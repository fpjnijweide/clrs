from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

from clrs._src.memory_models.ntm.ntm_memory import NTMMemory
from clrs._src.processors import _Fn, _Array, GATv2
from jax import nn

class Gatv2_NTM(GATv2):
    """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

    def __init__(
            self,
            out_size: int,
            nb_heads: int,
            mid_size: Optional[int] = None,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gatv2_ntm_aggr',
    ):
        super().__init__(out_size, nb_heads, mid_size, activation, residual, use_ln, name)
        self.beta_g_y_t_dense_layer = hk.Linear(output_size=3)
        self.memory_type = NTMMemory()
        self.s_t_dense_layer = hk.Linear(output_size=self.memory_type.memory_size)
        self.memory_state = None

        self.write_node_fts = None
        self.read_node_fts = None

        self.init_reset = False
        self.write_node_fts_layer = hk.initializers.RandomNormal(stddev=0.5)
        self.write_edge_fts_layer = hk.initializers.RandomNormal()
        self.read_edge_fts_layer = hk.initializers.RandomNormal()



        self.read_nodes_amount = 1
        self.write_nodes_amount = 5
        self.write_split_list = jnp.arange(1,self.write_nodes_amount+1)

    def __call__(
            self,
            node_fts: _Array,
            edge_fts: _Array,
            graph_fts: _Array,
            adj_mat: _Array,
            hidden: _Array,
            **unused_kwargs,
    ) -> _Array:
        """GATv2 inference step."""
        b, n, _ = node_fts.shape


        if not self.memory_state or len(self.memory_state) != b:
            self.memory_state = self.memory_type.initial_state(node_fts)
        # else:
        #     if not self.init_reset:
        #         self.memory_state = self.memory_type.initial_state(node_fts)
        #         self.init_reset = True
        #       We need to reset the memory state the second time we encounter this bit of code
        #       As the first bit was merely encountered during the haiku init


        extra_node_fts_shape = jnp.array(node_fts.shape)
        extra_node_fts_shape[0] = 1
        extra_node_fts_shape[1] = self.write_nodes_amount

        self.write_node_fts = self.write_node_fts_layer(shape=extra_node_fts_shape,dtype=jnp.float32)
        self.write_node_fts = jnp.repeat(self.write_node_fts,b,axis=0)

        if not self.read_node_fts:
            self.read_node_fts = jnp.average(node_fts,axis=1)

        # Setting the new node fts

        node_fts_new = jnp.concatenate([node_fts, self.read_node_fts,self.write_node_fts],axis=1)



        new_edge_feat_shape = jnp.array(edge_fts.shape)
        edge_fts_embedding_size = new_edge_feat_shape[-1]
        edge_fts_embedding_size_for_write = (self.write_nodes_amount,edge_fts_embedding_size)

        new_edge_feat_shape[1] = n+self.write_nodes_amount+self.read_nodes_amount
        new_edge_feat_shape[2] = n+self.write_nodes_amount+self.read_nodes_amount

        new_edge_fts = jnp.zeros(new_edge_feat_shape)
        new_edge_fts[:,:n,:n,:] = edge_fts

        # The edge features to/from write/read nodes will be generated via a dense layer

        write_edge_fts = self.write_edge_fts_layer(shape=edge_fts_embedding_size_for_write, dtype=jnp.float32)
        read_edge_fts = self.read_edge_fts_layer(shape=edge_fts_embedding_size, dtype=jnp.float32)

        write_edge_fts_tiled = jnp.tile(write_edge_fts,[b,n,1,1])
        read_edge_fts_tiled = jnp.tile(read_edge_fts,[b,self.read_nodes_amount,n,1])

        new_edge_fts[:,:n,n+self.read_nodes_amount:n+self.read_nodes_amount+self.write_nodes_amount,:] = write_edge_fts_tiled # From normal nodes to write nodes: edges
        new_edge_fts[:,n:n+self.read_nodes_amount,:n,:] = read_edge_fts_tiled # From read nodes to normal nodes: edges


        # Setting the ajdacency matrix

        adj_mat_new = jnp.zeros([b,n+self.write_nodes_amount+self.read_nodes_amount,n+self.write_nodes_amount+self.read_nodes_amount])
        adj_mat_new[:,:n,:n] = adj_mat

        adj_mat_new[:,:n,n+self.read_nodes_amount:n+self.read_nodes_amount+self.write_nodes_amount] = 1 # From normal nodes to write nodes: edges
        adj_mat_new[:,n:n+self.read_nodes_amount,:n] = 1 # From read nodes to normal nodes: edges


        # assert adj_mat.shape == (b, n, n)



        # TODO what is hidden?

        ################################################
        # Network is called here
        concatenated_ret = super().__call__(node_fts_new, new_edge_fts, graph_fts, adj_mat_new, hidden, **unused_kwargs)
        ################################################

        real_ret,read_ret,write_ret = jnp.split(concatenated_ret,[n,n+self.read_nodes_amount],axis=1)

        s_t_pre_dense,beta_g_y_t,k_t,a_t,e_t = jnp.split(write_ret,self.write_nodes_amount,axis=1)



        s_t = self.s_t_dense_layer(s_t_pre_dense.squeeze())
        beta_g_y_t_post_dense = self.beta_g_y_t_dense_layer(beta_g_y_t.squeeze())

        beta_t,g_t,gamma_t = jnp.split(beta_g_y_t_post_dense,3,axis=1)
        k = jnp.tanh(k_t.squeeze())
        beta = nn.softplus(beta_t)
        g = nn.sigmoid(g_t)
        s = nn.softmax(
            s_t
        )
        gamma = nn.softplus(gamma_t) + 1


        e_t = e_t.squeeze()
        a_t = a_t.squeeze()
        # head_parameter_list is a list for each read/write head

        memory_inputs_list = []
        # TODO use concatenated inputs instead
        # TODO memory state already has batch dimensions! also check inner function for this
        concatenated_inputs = jnp.concatenate(k,beta,g,s,gamma,e_t,a_t,axis=1)
        for i in range(b):

            head_parameters = (k[i],beta[i],g[i],s[i],gamma[i])
            erase_add_tuple = (e_t[i],a_t[i])

            head_parameter_list = [head_parameters] # TODO adapt for multiple heads
            erase_add_list = [erase_add_tuple] # TODO adapt for multiple heads
            final_inputs = (head_parameter_list, erase_add_list)
            memory_inputs_list.append(final_inputs)

        NTM_read_vector_lists = []
        next_states = []
        ################################################
        # Memory is called here
        for i in range(b):
            memory_state=self.memory_state[i]
            memory_inputs=memory_inputs_list[i]
            NTM_read_vector_list_here, next_state_here = self.memory_type(memory_inputs,memory_state)
            NTM_read_vector_lists.append(NTM_read_vector_list_here[0]) # TODO adapt for multiple heads
            next_states.append(next_state_here)

        ################################################


        self.read_node_fts = jnp.concatenate(NTM_read_vector_lists,axis=0)
        self.memory_state = next_states



        # TODO remove for loops
        # TODO how to refresh memory for each new example set? the network will be called multiple times until done
        # TODO ensure there is no dense layer for itself?




        # TODO do for all network types
        # TODO do for all memory types

        # TODO experiment with memory size

        return real_ret

