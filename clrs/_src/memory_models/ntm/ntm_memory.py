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


# TODO how to make sure haiku init with dummy variable doesn't ruin M with worthless writes?

# NTMControllerState = collections.namedtuple('NTMControllerState',
#                                             ('controller_state', 'read_vector_list', 'w_list', 'M'))

class NTMState(NamedTuple):
    M: jnp.ndarray
    w_list: List[jnp.ndarray]
    read_vector_list: List[jnp.ndarray]


class NTMMemory(hk.RNNCore):
    def __init__(self, memory_vector_dim=None, memory_size=20, read_head_num=1, write_head_num=1,
                 addressing_mode='content_and_location', shift_range=1, clip_value=20, init_mode='constant',name: Optional[str] = None):
        # self.controller_layers = controller_layers
        # self.controller_units = controller_units
        super().__init__(name=name)
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.read_head_num = read_head_num
        self.write_head_num = write_head_num
        self.addressing_mode = addressing_mode
        # super(NTMCell2, self).__init__(**kwargs)

        # def single_cell(num_units):
        #     return keras.layers.LSTMCell(num_units, unit_forget_bias=True)

        # self.controller = keras.layers.StackedRNNCells(
        #     [single_cell(self.controller_units) for _ in range(self.controller_layers)])

        self.clip_value = clip_value
        self.init_mode = init_mode

        # self.step = 0

        # self.output_dim = output_dim
        self.shift_range = shift_range

        self.beta_g_y_t_dense_layer = hk.Linear(output_size=3)
        self.s_t_dense_layer = hk.Linear(output_size=self.memory_size)

        self.total_heads = self.memory.write_head_num + self.memory.read_head_num
        self.w_nodes_amount = 3 * self.total_heads
        self.erase_add_nodes_amount = 2 * (self.memory.write_head_num)

        self.read_nodes_amount = self.memory.read_head_num
        self.write_nodes_amount = self.w_nodes_amount + self.erase_add_nodes_amount
        # self.num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        # self.num_heads = self.read_head_num + self.write_head_num
        # self.total_parameter_num = self.num_parameters_per_head * self.num_heads +\
        # self.memory_vector_dim * 2 * self.write_head_num

        # self.controller_proj_initializer = create_linear_initializer(self.controller_units)
        # self.output_proj_initializer = create_linear_initializer(
        #     self.controller_units + self.memory_vector_dim * self.read_head_num)
        # self.controller_proj = keras.layers.Dense(self.total_parameter_num, activation=None,
        #                                           kernel_initializer=self.controller_proj_initializer)
        # self.output_proj = keras.layers.Dense(output_dim, activation=None,
        #                                       kernel_initializer=self.output_proj_initializer)
        # self._get_init_state_vars()

    def initial_state(self, node_fts):
        batch_size, n, h = node_fts.shape
        self.memory_vector_dim = h

        read_vector_list = []
        for i in range(self.read_head_num):
            new_read_vector = jnp.tanh(hk.get_parameter(f"NTM_read_vector_{i}", shape=[self.memory_vector_dim, ],
                                                        init=hk.initializers.VarianceScaling(1.0, "fan_avg",
                                                                                             "uniform")))
            read_vector_list.append(new_read_vector)

        w_var_list = []
        for i in range(self.read_head_num + self.write_head_num):
            new_w_var = nn.softmax(hk.get_parameter(f"NTM_w_{i}", shape=[self.memory_size, ],
                                                    init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")))
            w_var_list.append(new_w_var)

        if self.init_mode == 'learned':
            M = jnp.tanh(hk.get_parameter("NTM_M", shape=[self.memory_size, self.memory_vector_dim],
                                          init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")))
        elif self.init_mode == 'random':
            M = jnp.tanh(
                hk.initializers.RandomNormal(stddev=0.5)([self.memory_size, self.memory_vector_dim], dtype=jnp.float32))
        elif self.init_mode == 'constant':
            M = jnp.full([self.memory_size, self.memory_vector_dim], 1e-6)
        else:
            raise RuntimeError("invalid init mode")

        state = NTMState(M, w_var_list, read_vector_list)
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state

    def __call__(self, inputs, prev_state: NTMState):
        # head_parameter_list, erase_add_list, prev_M, prev_w_list

        head_parameter_list, erase_add_list = inputs
        prev_M = prev_state.M
        prev_w_list = prev_state.w_list

        # prev_read_vector_list = prev_state[1]
        #
        # controller_input = jnp.concatenate([x] + prev_read_vector_list, axis=1)
        # controller_output, controller_state = self.controller(controller_input, prev_state[0])
        #
        # parameters = self.controller_proj(controller_output)
        # head_parameter_list = jnp.split(parameters[:, :self.num_parameters_per_head * self.num_heads], self.num_heads,
        #                                axis=1)
        # erase_add_list = jnp.split(parameters[:, self.num_parameters_per_head * self.num_heads:],
        #                           2 * self.write_head_num, axis=1)

        # prev_w_list = prev_state[2]
        # prev_M = prev_state[3]
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

    def _addressing(self, k, beta, g, s, gamma, prev_M, prev_w):

        # Sec 3.3.1 Focusing by Content

        # Cosine Similarity

        k = jnp.expand_dims(k, axis=2)
        inner_product = jnp.matmul(prev_M, k)
        k_norm = jnp.sqrt(jnp.sum(jnp.square(k), axis=1, keepdims=True))
        M_norm = jnp.sqrt(jnp.sum(jnp.square(prev_M), axis=2, keepdims=True))
        norm_product = M_norm * k_norm
        K = jnp.squeeze(inner_product / (norm_product + 1e-8))  # eq (6)

        # Calculating w^c

        K_amplified = jnp.exp(beta * K)
        w_c = K_amplified / jnp.sum(K_amplified, axis=1, keepdims=True)  # eq (5)

        if self.addressing_mode == 'content':  # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        # g = jnp.expand_dims(g, axis=1)




        # # k = jnp.expand_dims(k, axis=1)
        # inner_product = jnp.matmul(prev_M, k)
        # k_norm = jnp.sqrt(jnp.sum(jnp.square(k), keepdims=True))
        # M_norm = jnp.sqrt(jnp.sum(jnp.square(prev_M), axis=1, keepdims=True))
        # norm_product = M_norm * k_norm
        # K = jnp.squeeze(inner_product / (jnp.squeeze(norm_product) + 1e-8))  # eq (6)
        #
        # # Calculating w^c
        #
        # K_amplified = jnp.exp(beta * K)
        # w_c = K_amplified / jnp.sum(K_amplified)  # eq (5)
        #
        # if self.addressing_mode == 'content':  # Only focus on content
        #     return w_c
        #
        # # Sec 3.3.2 Focusing by Location
        #
        # # g = jnp.expand_dims(g, axis=1)





        w_g = g * w_c + (1 - g) * prev_w  # eq (7)

        s = jnp.concatenate([s[:, :self.shift_range + 1],
                             jnp.zeros([s.shape[0], self.memory_size - (self.shift_range * 2 + 1)]),
                             s[:, -self.shift_range:]], axis=1)
        t = jnp.concatenate([jnp.flip(s, axis=1), jnp.flip(s, axis=1)], axis=1)
        s_matrix = jnp.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)], axis=1)
        w_ = jnp.sum(jnp.expand_dims(w_g, axis=1) * s_matrix, axis=2)  # eq (8)
        w_sharpen = jnp.power(w_, gamma)
        w = w_sharpen / jnp.sum(w_sharpen, axis=1, keepdims=True)  # eq (9)

        return w

    def prepare_memory_input(self, concatenated_ret, n):
        # TODO reduce for loops?

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

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #     # with jnp.variable_scope('init', reuse=self.reuse):
    #     read_vector_list = [expand(jnp.tanh(var), dim=0, N=batch_size) for var in self.read_vector_var_list]
    #
    #     w_list = [expand(nn.softmax(var), dim=0, N=batch_size) for var in self.w_var_list]
    #
    #     # controller_init_state = self.controller.get_initial_state(inputs=None, batch_size=batch_size, dtype=dtype)
    #
    #     M = expand(self.M_var, dim=0, N=batch_size)
    #
    #     return NTMControllerState(
    #         # controller_state=controller_init_state,
    #         read_vector_list=read_vector_list,
    #         w_list=w_list,
    #         M=M)

    # @property
    # def state_size(self):
    #     return NTMControllerState(
    #         controller_state=self.controller.state_size[0],
    #         read_vector_list=[self.memory_vector_dim for _ in range(self.read_head_num)],
    #         w_list=[self.memory_size for _ in range(self.read_head_num + self.write_head_num)],
    #         M=jnp.TensorShape([self.memory_size, self.memory_vector_dim]))

