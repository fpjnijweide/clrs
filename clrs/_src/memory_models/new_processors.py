from typing import Optional

import jax

from clrs._src.memory_models.ntm.ntm_memory import NTM_memory
from clrs._src.processors import _Fn, _Array, GATv2


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
            name: str = 'gatv2_aggr',
    ):
        super().__init__(out_size, nb_heads, mid_size, activation, residual, use_ln, name)
        self.memory = NTM_memory()
        self.ntm_state = None

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

        if not self.ntm_state:
            self.ntm_state = self.memory.initial_state(node_fts)

        # TODO probably have to use get and set state. saving as field here will lead to issues
        # because it will already have written to memory after init

        # TODO add read/wrrite nodes here
        #
        # TODO ensure its adjacency matrix has 0s for read/write node?
        # TODO ensure there is no dense layer for itself?
        # TODO end of read/write node adding
        # TODO set the adj mat here

        # call the parent network
        ret = super().__call__(node_fts, edge_fts, graph_fts, adj_mat, hidden, **unused_kwargs)

        # TODO remove read write nodes and such
        # TODO execute some dense layers
        # TODO stack variables to proper size
        # TODO call NTM

        return ret

# TODO transfer NTM to all others, including gatv1

# TODO GATV2 deque
# TODO transfer deque to all others

# TODO PGN memory nodes and gat memory nodes
