# This file is edited from https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/ntm.py
# Which is the source code behind the "Implementing Neural Turing Machines" paper
# which I cite in my thesis
# I also adapted many lines from https://git.droidware.info/wchen342/NeuralTuringMachine/src/branch/master/ntm.py
# which is a tensorflow 2.0 port of that same code (as I am entirely unfamiliar with TF1.0 and
# struggled to find documentation for some parts of the code that I could not execute)
#
# all I did was port this code to JAX/haiku and remove the RNN/LSTM parts
# I take no credit for any of the ideas
# - Frederik Nijweide, June 2022
# -----------------------
from typing import List, NamedTuple, Optional

import haiku as hk
import jax.numpy as jnp
# noinspection PyProtectedMember
from haiku._src.recurrent import add_batch
from jax import nn


class StorageNodes(hk.RNNCore):
    def __init__(self, memory_size=20, read_head_num=1, write_head_num=1, name: Optional[str] = None):
        super().__init__(name=name)
        self.memory_size = memory_size
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num


        # self.beta_g_y_t_dense_layer = hk.Linear(output_size=3)
        # self.s_t_dense_layer = hk.Linear(output_size=self.memory_size)

        self.total_heads = self.write_head_num + self.read_head_num

        self.read_nodes_amount = self.read_head_num
        self.write_nodes_amount = self.write_head_num

    def initial_state(self, node_fts):
        batch_size, n, h = node_fts.shape
        self.embedding_size = h

        # TODO init these nodes and edges via layer as well!
        state = jnp.zeros([batch_size,self.memory_size,h])
        return state

    def __call__(self, inputs, prev_state: jnp.array):

        head_parameter_list, erase_add_list = inputs
        prev_M = prev_state.M
        prev_w_list = prev_state.w_list


        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k, beta, g, s, gamma = head_parameter
            w = self._addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = jnp.sum(jnp.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i in range(self.write_head_num):
            e_t,a_t = erase_add_list[i]
            erase_vector = jnp.expand_dims(nn.sigmoid(e_t), axis=1)
            add_vector = jnp.expand_dims(nn.tanh(a_t), axis=1)
            w = jnp.expand_dims(write_w_list[i], axis=2)
            M = M * (jnp.ones(M.shape) - jnp.matmul(w, erase_vector)) + jnp.matmul(w, add_vector)

        # NTM_output = self.output_proj(jnp.concatenate([controller_output] + read_vector_list, axis=1))
        for i, read_vector in enumerate(read_vector_list):
            read_vector_list[i] = jnp.clip(read_vector_list[i], -self.clip_value, self.clip_value)

        # self.step += 1

        next_state = NTMState(M, w_list, read_vector_list)
        output = jnp.stack(read_vector_list,axis=1)

        return output, next_state

    # def _addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
    #
    #     # Sec 3.3.1 Focusing by Content
    #
    #     # Cosine Similarity
    #
    #     k = jnp.expand_dims(k, axis=2)
    #     inner_product = jnp.matmul(prev_M, k)
    #     k_norm = jnp.sqrt(jnp.sum(jnp.square(k), axis=1, keepdims=True))
    #     M_norm = jnp.sqrt(jnp.sum(jnp.square(prev_M), axis=2, keepdims=True))
    #     norm_product = M_norm * k_norm
    #     K = jnp.squeeze(inner_product / (norm_product + 1e-8))  # eq (6)
    #
    #     # Calculating w^c
    #
    #     K_amplified = jnp.exp(beta * K)
    #     w_c = K_amplified / jnp.sum(K_amplified, axis=1, keepdims=True)  # eq (5)
    #
    #     if self.addressing_mode == 'content':  # Only focus on content
    #         return w_c
    #
    #     w_g = g * w_c + (1 - g) * prev_w  # eq (7)
    #
    #     s = jnp.concatenate([s[:, :self.shift_range + 1],
    #                          jnp.zeros([s.shape[0], self.memory_size - (self.shift_range * 2 + 1)]),
    #                          s[:, -self.shift_range:]], axis=1)
    #     t = jnp.concatenate([jnp.flip(s, axis=1), jnp.flip(s, axis=1)], axis=1)
    #     s_matrix = jnp.stack(
    #         [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)], axis=1)
    #     w_ = jnp.sum(jnp.expand_dims(w_g, axis=1) * s_matrix, axis=2)  # eq (8)
    #     w_sharpen = jnp.power(w_, gamma)
    #     w = w_sharpen / jnp.sum(w_sharpen, axis=1, keepdims=True)  # eq (9)
    #
    #     return w

    def prepare_memory_input(self, concatenated_ret, n):
        real_ret, _, write_ret = jnp.split(concatenated_ret, [n, n + self.read_head_num], axis=1)
        w_nodes, a_e_nodes = jnp.split(write_ret, [self.w_nodes_amount], axis=1)
        list_of_params_for_each_w = jnp.split(w_nodes, self.write_head_num+self.read_head_num, axis=1)
        list_of_params_for_each_write = jnp.split(a_e_nodes, self.write_head_num, axis=1)
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
        return final_tuple, real_ret

