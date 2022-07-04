# This file is edited from https://github.com/MarkPKCollier/NeuralTuringMachine/blob/master/utils.py
# Which is the source code behind the "Implementing Neural Turing Machines" paper
# which I cite in my thesis
# I also adapted many lines from https://git.droidware.info/wchen342/NeuralTuringMachine/src/branch/master/utils.py
# which is a tensorflow 2.0 port of that same code (as I am entirely unfamiliar with TF1.0 and
# struggled to find documentation for some parts of the code that I could not execute)
# -----------------------
# import numpy as np
import jax.numpy as jnp


# def expand(x, dim, N):
#     return jnp.concatenate([jnp.expand_dims(x, dim) for _ in range(N)], axis=dim)
#
# def learned_init(units):
#     # TODO fully connected layer
#     return jnp.squeeze(jnp.contrib.layers.fully_connected(jnp.ones([1, 1]), units,
#         activation_fn=None, biases_initializer=None))
#
# def create_linear_initializer(input_size, dtype=jnp.float32):
#     stddev = 1.0 / jnp.sqrt(input_size)
#     return jnp.truncated_normal_initializer(stddev=stddev, dtype=dtype)


def expand(x, dim, N):
    ndim = jnp.shape(jnp.shape(x))[0]
    expand_idx = jnp.concatenate([jnp.ones((jnp.maximum(0, dim),), dtype=jnp.int32), jnp.reshape(N, (-1,)),
                                  jnp.ones((jnp.minimum(ndim - dim, ndim),), dtype=jnp.int32)], axis=0)
    return jnp.tile(jnp.expand_dims(x, dim), expand_idx)

# def learned_init(units):
#     return jnp.Variable(initial_value=keras.initializers.glorot_uniform()(shape=(units,)))
#
#
# def create_linear_initializer(input_size, dtype=jnp.float32):
#     stddev = 1.0 / jnp.sqrt(input_size)
#     return jnp.truncated_normal(stddev=stddev, dtype=dtype)
