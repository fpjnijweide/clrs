# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from jax import numpy as jnp


def extend_features(adj_mat, edge_fts, hidden, node_fts,read_nodes_amount,write_nodes_amount,write_node_fts_layer,write_edge_fts_layer,read_node_fts,read_edge_fts_layer,read_node_fts_layer):
    b, n, _ = node_fts.shape
    write_node_fts = write_node_fts_layer(shape=(1,write_nodes_amount,node_fts.shape[2]), dtype=jnp.float32)
    write_node_fts = jnp.repeat(write_node_fts, b, axis=0)
    if read_node_fts is None:
        read_node_fts = read_node_fts_layer(shape=(1,read_nodes_amount,node_fts.shape[2]), dtype=jnp.float32)
        read_node_fts = jnp.repeat(read_node_fts, b, axis=0)
    # Setting the new node fts
    node_fts_new = jnp.concatenate([node_fts, read_node_fts, write_node_fts], axis=1)
    edge_fts_embedding_size = edge_fts.shape[-1]
    edge_fts_embedding_size_for_write = (write_nodes_amount, edge_fts_embedding_size)
    new_n = n + write_nodes_amount + read_nodes_amount
    new_edge_fts = jnp.zeros(shape=(b,new_n,new_n,edge_fts_embedding_size))
    new_edge_fts = new_edge_fts.at[:, :n, :n, :].set(edge_fts)
    # The edge features to/from write/read nodes will be generated via a dense layer
    write_edge_fts = write_edge_fts_layer(shape=edge_fts_embedding_size_for_write, dtype=jnp.float32)
    read_edge_fts = read_edge_fts_layer(shape=(1, edge_fts_embedding_size), dtype=jnp.float32)
    write_edge_fts_tiled = jnp.tile(write_edge_fts, [b, n, 1, 1])
    read_edge_fts_tiled = jnp.tile(read_edge_fts, [b, read_nodes_amount, n, 1])
    new_edge_fts = new_edge_fts.at[:, :n,
                   n + read_nodes_amount:n + read_nodes_amount + write_nodes_amount, :].set(
        write_edge_fts_tiled)  # From normal nodes to write nodes: edges
    new_edge_fts = new_edge_fts.at[:, n:n + read_nodes_amount, :n, :].set(
        read_edge_fts_tiled)  # From read nodes to normal nodes: edges
    # Setting the ajdacency matrix
    adj_mat_new = jnp.zeros([b, n + write_nodes_amount + read_nodes_amount,
                             n + write_nodes_amount + read_nodes_amount])
    adj_mat_new = adj_mat_new.at[:, :n, :n].set(adj_mat)
    adj_mat_new = adj_mat_new.at[:, :n,
                  n + read_nodes_amount:n + read_nodes_amount + write_nodes_amount].set(
        1)  # From normal nodes to write nodes: edges
    adj_mat_new = adj_mat_new.at[:, n:n + read_nodes_amount, :n].set(
        1)  # From read nodes to normal nodes: edges
    # hidden_new_shape = jnp.array(hidden.shape)
    # hidden_new_shape = hidden_new_shape.at[1].set(
    #     hidden_new_shape[1] + read_nodes_amount + write_nodes_amount)
    hidden_new = jnp.zeros((hidden.shape[0],hidden.shape[1]+read_nodes_amount + write_nodes_amount,hidden.shape[2]), dtype=jnp.float32)
    hidden_new = hidden_new.at[:, :n, :].set(hidden)  # TODO what to do about hidden
    return adj_mat_new, hidden_new, new_edge_fts, node_fts_new
