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
#-----------------------



import jax.numpy as jnp
from jax import nn
import collections
from ntm_utils import expand, learned_init, create_linear_initializer


# TODO clip parameters
# TODO freeze the gradient on M and such
# TOOD init parameters

NTMControllerState = collections.namedtuple('NTMControllerState',
                                            ('controller_state', 'read_vector_list', 'w_list', 'M'))


class NTM_memory():
    def __init__(self, memory_size=20, read_head_num=1,
                 write_head_num=1,
                 addressing_mode='content_and_location', shift_range=1, clip_value=20,
                 init_mode='constant'):
        # self.controller_layers = controller_layers
        # self.controller_units = controller_units
        self.memory_size = memory_size
        # self.memory_vector_dim = memory_vector_dim
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
        # self.num_parameters_per_head = self.memory_vector_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1
        # self.num_heads = self.read_head_num + self.write_head_num
        # self.total_parameter_num = self.num_parameters_per_head * self.num_heads + self.memory_vector_dim * 2 * self.write_head_num

        # self.controller_proj_initializer = create_linear_initializer(self.controller_units)
        # self.output_proj_initializer = create_linear_initializer(
        #     self.controller_units + self.memory_vector_dim * self.read_head_num)
        # self.controller_proj = keras.layers.Dense(self.total_parameter_num, activation=None,
        #                                           kernel_initializer=self.controller_proj_initializer)
        # self.output_proj = keras.layers.Dense(output_dim, activation=None,
        #                                       kernel_initializer=self.output_proj_initializer)
        self._get_init_state_vars()

    def call(self, head_parameter_list, head_erase_add_list, prev_M,prev_w_list):
        # prev_read_vector_list = prev_state[1]
        #
        # controller_input = jnp.concatenate([x] + prev_read_vector_list, axis=1)
        # controller_output, controller_state = self.controller(controller_input, prev_state[0])
        #
        # parameters = self.controller_proj(controller_output)
        # parameters = jnp.clip(parameters, -self.clip_value, self.clip_value)
        # head_parameter_list = jnp.split(parameters[:, :self.num_parameters_per_head * self.num_heads], self.num_heads,
        #                                axis=1)
        # erase_add_list = jnp.split(parameters[:, self.num_parameters_per_head * self.num_heads:],
        #                           2 * self.write_head_num, axis=1)

        # prev_w_list = prev_state[2]
        # prev_M = prev_state[3]
        w_list = []
        for i, head_parameter in enumerate(head_parameter_list):
            k = jnp.tanh(head_parameter[0])
            beta = nn.softplus(head_parameter[1])
            g = nn.sigmoid(head_parameter[2])
            s = nn.softmax(
                head_parameter[3]
            )
            gamma = nn.softplus(head_parameter[4]) + 1
            w = self._addressing(k, beta, g, s, gamma, prev_M, prev_w_list[i])
            w_list.append(w)

        # Reading (Sec 3.1)

        read_w_list = w_list[:self.read_head_num]
        read_vector_list = []
        for i in range(self.read_head_num):
            read_vector = jnp.sum(jnp.expand_dims(read_w_list[i], axis=2) * prev_M, axis=1)
            read_vector_list.append(read_vector)

        # Writing (Sec 3.2)

        write_w_list = w_list[self.read_head_num:]
        M = prev_M
        for i, er_add_list in enumerate(head_erase_add_list):
            w = jnp.expand_dims(write_w_list[i], axis=2)
            erase_vector = jnp.expand_dims(nn.sigmoid(er_add_list[0]), axis=1)
            add_vector = jnp.expand_dims(jnp.tanh(er_add_list[1]), axis=1)
            M = M * (jnp.ones(M.get_shape()) - jnp.matmul(w, erase_vector)) + jnp.matmul(w, add_vector)

        # NTM_output = self.output_proj(jnp.concatenate([controller_output] + read_vector_list, axis=1))
        # NTM_output = jnp.clip(NTM_output, -self.clip_value, self.clip_value)

        # self.step += 1
        return read_vector_list,M,w_list #, NTMControllerState(controller_state=controller_state, read_vector_list=read_vector_list, w_list=w_list, M=M)

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

        K_amplified = jnp.exp(jnp.expand_dims(beta, axis=1) * K)
        w_c = K_amplified / jnp.sum(K_amplified, axis=1, keepdims=True)  # eq (5)

        if self.addressing_mode == 'content':  # Only focus on content
            return w_c

        # Sec 3.3.2 Focusing by Location

        g = jnp.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w  # eq (7)

        s = jnp.concatenate([s[:, :self.shift_range + 1],
                       jnp.zeros([s.get_shape()[0], self.memory_size - (self.shift_range * 2 + 1)]),
                       s[:, -self.shift_range:]], axis=1)
        t = jnp.concatenate([jnp.flip(s, axis=1), jnp.flip(s, axis=1)], axis=1)
        s_matrix = jnp.stack(
            [t[:, self.memory_size - i - 1:self.memory_size * 2 - i - 1] for i in range(self.memory_size)], axis=1)
        w_ = jnp.sum(jnp.expand_dims(w_g, axis=1) * s_matrix, axis=2)  # eq (8)
        w_sharpen = jnp.power(w_, jnp.expand_dims(gamma, axis=1))
        w = w_sharpen / jnp.sum(w_sharpen, axis=1, keepdims=True)  # eq (9)

        return w

    def _get_init_state_vars(self):
        self.read_vector_var_list = [self.add_variable('read_vector_{}'.format(i), [self.memory_vector_dim, ],
                                                       initializer=keras.initializers.glorot_uniform()) for i in
                                     range(self.read_head_num)]
        self.w_var_list = [self.add_variable('w_{}'.format(i), [self.memory_size, ],
                                             initializer=keras.initializers.glorot_uniform()) for i in
                           range(self.read_head_num + self.write_head_num)]
        if self.init_mode == 'learned':
            pass
            # self.M_var = jnp.tanh(self.add_variable('Memory', [self.memory_size, self.memory_vector_dim, ],
            #                                        initializer=keras.initializers.glorot_uniform()))
        elif self.init_mode == 'random':
            pass
            # self.M_var = jnp.tanh(self.add_variable('Memory', [self.memory_size, self.memory_vector_dim],
            #                                        initializer=jnp.random_normal_initializer(mean=0.0, stddev=0.5)))
        elif self.init_mode == 'constant':
            self.M_var = self.add_variable('Memory', [self.memory_size, self.memory_vector_dim],
                                           initializer=jnp.constant_initializer(1e-6))

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

