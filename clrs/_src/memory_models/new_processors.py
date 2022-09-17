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
            name: str = 'gatv2_ntm',
    ):
        super().__init__(out_size, nb_heads, mid_size, activation, residual, use_ln, name)
        self.use_lstm=True
        self.beta_g_y_t_dense_layer = hk.Linear(output_size=3)
        self.memory_type = NTMMemory()
        self.s_t_dense_layer = hk.Linear(output_size=self.memory_type.memory_size)
        self.lstm_state = None

        self.write_node_fts = None
        self.read_node_fts = None

        self.init_reset = False
        self.write_node_fts_layer = hk.initializers.RandomNormal(stddev=0.5)
        self.read_node_fts_layer = hk.initializers.RandomNormal(stddev=0.5)
        self.write_edge_fts_layer = hk.initializers.RandomNormal()
        self.read_edge_fts_layer = hk.initializers.RandomNormal()

        self.read_nodes_amount = self.memory_type.read_head_num
        self.total_heads = self.memory_type.write_head_num + self.memory_type.read_head_num
        self.w_nodes_amount = 3 * self.total_heads
        self.erase_add_nodes_amount = 2 * (self.memory_type.write_head_num)
        self.write_nodes_amount = self.w_nodes_amount + self.erase_add_nodes_amount
        self.write_split_list = jnp.arange(1, self.write_nodes_amount + 1)

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

        if not self.lstm_state:
            self.lstm_state = self.memory_type.initial_state(node_fts)
        # else:
        #     if not self.init_reset:
        #         self.memory_state = self.memory_type.initial_state(node_fts)
        #         self.init_reset = True
        #       We need to reset the memory state the second time we encounter this bit of code
        #       As the first bit was merely encountered during the haiku init

        write_node_fts_shape = jnp.array(node_fts.shape)
        write_node_fts_shape = write_node_fts_shape.at[0].set(1)
        write_node_fts_shape = write_node_fts_shape.at[1].set(self.write_nodes_amount)

        self.write_node_fts = self.write_node_fts_layer(shape=write_node_fts_shape, dtype=jnp.float32)
        self.write_node_fts = jnp.repeat(self.write_node_fts, b, axis=0)

        if self.read_node_fts is None:
            read_node_fts_shape = jnp.array(node_fts.shape)
            read_node_fts_shape = read_node_fts_shape.at[0].set(1)
            read_node_fts_shape = read_node_fts_shape.at[1].set(self.read_nodes_amount)

            self.read_node_fts = self.write_node_fts_layer(shape=read_node_fts_shape, dtype=jnp.float32)
            self.read_node_fts = jnp.repeat(self.read_node_fts, b, axis=0)

        # Setting the new node fts

        node_fts_new = jnp.concatenate([node_fts, self.read_node_fts, self.write_node_fts], axis=1)

        new_edge_feat_shape = jnp.array(edge_fts.shape)
        edge_fts_embedding_size = new_edge_feat_shape[-1]
        edge_fts_embedding_size_for_write = (self.write_nodes_amount, edge_fts_embedding_size)

        new_edge_feat_shape = new_edge_feat_shape.at[1].set(n + self.write_nodes_amount + self.read_nodes_amount)
        new_edge_feat_shape = new_edge_feat_shape.at[2].set(n + self.write_nodes_amount + self.read_nodes_amount)

        new_edge_fts = jnp.zeros(new_edge_feat_shape)
        new_edge_fts = new_edge_fts.at[:, :n, :n, :].set(edge_fts)

        # The edge features to/from write/read nodes will be generated via a dense layer

        write_edge_fts = self.write_edge_fts_layer(shape=edge_fts_embedding_size_for_write, dtype=jnp.float32)
        read_edge_fts = self.read_edge_fts_layer(shape=(1,edge_fts_embedding_size), dtype=jnp.float32)

        write_edge_fts_tiled = jnp.tile(write_edge_fts, [b, n, 1, 1])
        read_edge_fts_tiled = jnp.tile(read_edge_fts, [b, self.read_nodes_amount, n, 1])

        new_edge_fts = new_edge_fts.at[:, :n,
                       n + self.read_nodes_amount:n + self.read_nodes_amount + self.write_nodes_amount, :].set(
            write_edge_fts_tiled)  # From normal nodes to write nodes: edges
        new_edge_fts = new_edge_fts.at[:, n:n + self.read_nodes_amount, :n, :].set(
            read_edge_fts_tiled)  # From read nodes to normal nodes: edges

        # Setting the ajdacency matrix

        adj_mat_new = jnp.zeros([b, n + self.write_nodes_amount + self.read_nodes_amount,
                                 n + self.write_nodes_amount + self.read_nodes_amount])
        adj_mat_new = adj_mat_new.at[:, :n, :n].set(adj_mat)

        adj_mat_new = adj_mat_new.at[:, :n,
                      n + self.read_nodes_amount:n + self.read_nodes_amount + self.write_nodes_amount].set(
            1)  # From normal nodes to write nodes: edges
        adj_mat_new = adj_mat_new.at[:, n:n + self.read_nodes_amount, :n].set(
            1)  # From read nodes to normal nodes: edges

        hidden_new_shape = jnp.array(hidden.shape)
        hidden_new_shape = hidden_new_shape.at[1].set(hidden_new_shape[1]+self.read_nodes_amount+self.write_nodes_amount)
        hidden_new = jnp.zeros(hidden_new_shape,dtype=jnp.float32)
        hidden_new = hidden_new.at[:,:n,:].set(hidden) # TODO what to do about hidden
        # assert adj_mat.shape == (b, n, n)

        ################################################
        # Network is called here
        concatenated_ret = super().__call__(node_fts_new, new_edge_fts, graph_fts, adj_mat_new, hidden_new, **unused_kwargs)
        ################################################

        real_ret, _, write_ret = jnp.split(concatenated_ret, [n, n + self.read_nodes_amount], axis=1)

        w_nodes, a_e_nodes = jnp.split(write_ret, [self.w_nodes_amount], axis=1)
        list_of_params_for_each_w = jnp.split(w_nodes, self.total_heads, axis=1)
        list_of_params_for_each_write = jnp.split(a_e_nodes, self.memory_type.write_head_num, axis=1)

        w_params_for_batch = []
        for i, params in enumerate(list_of_params_for_each_w):
            s_t_pre_dense, beta_g_y_t, k_t = jnp.split(params, 3, axis=1)
            s_t = self.s_t_dense_layer(s_t_pre_dense.squeeze())
            beta_g_y_t_post_dense = self.beta_g_y_t_dense_layer(beta_g_y_t.squeeze())

            beta_t, g_t, gamma_t = jnp.split(beta_g_y_t_post_dense, 3, axis=1)
            k = jnp.tanh(k_t.squeeze())
            beta = nn.softplus(beta_t)
            g = nn.sigmoid(g_t)
            s = nn.softmax(
                s_t
            )
            gamma = nn.softplus(gamma_t) + 1
            head_parameters = (k, beta, g, s, gamma)
            w_params_for_batch.append(head_parameters)

        e_a_params_for_batch = []
        for i, params in enumerate(list_of_params_for_each_write):
            e_t, a_t = jnp.split(params, 2, axis=1)
            e_t = e_t.squeeze()
            a_t = a_t.squeeze()
            e_a_final = (e_t, a_t)
            e_a_params_for_batch.append(e_a_final)

        final_tuple = (w_params_for_batch, e_a_params_for_batch)

        NTM_read_vector_lists, self.lstm_state = self.memory_type(final_tuple,
                                                                  self.lstm_state)
        self.read_node_fts = jnp.stack(NTM_read_vector_lists,axis=1)

        # TODO "lstm" state via baselines,etc
        # TODO how to refresh memory, read node fts for each new example set? the network will be called multiple times until done

        # TODO do for MPNN
        # TODO do for PGN?
        # TODO do for deque
        # TODO do for own architecture
        # TODO do for DNC



        # TODO experiment with memory size

        return real_ret


if __name__ == '__main__':
    net = Gatv2_NTM(1, 1)
    key = jax.random.PRNGKey(0)
    b = 4
    n = 8
    h_node = 3
    h_edge = 2
    h_graph = 1
    node_fts_shape = (b, n, h_node)
    edge_fts_shape = (b, n, n, h_edge)
    adj_mat_shape = (b, n, n)
    node_fts = jax.random.normal(key, node_fts_shape, dtype=jnp.float32)
    edge_fts = jax.random.normal(key, edge_fts_shape, dtype=jnp.float32)
    adj_mat = jax.random.randint(key, adj_mat_shape, 0, 1, dtype=jnp.int32)
