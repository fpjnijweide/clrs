from typing import Callable, Optional

from clrs._src.memory_models.new_processors import Gatv2_NTM
from clrs._src.processors import Processor, DeepSets, GAT, GATFull, GATv2, GATv2Full, MemNetFull, MemNetMasked, MPNN, \
    PGN, PGNMask

ProcessorFactory = Callable[[int], Processor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_heads: Optional[int] = None) -> ProcessorFactory:
  """Returns a processor factory.

  Args:
    kind: One of the available types of processor.
    use_ln: Whether the processor passes the output through a layernorm layer.
    nb_heads: Number of attention heads for GAT processors.
  Returns:
    A callable that takes an `out_size` parameter (equal to the hidden
    dimension of the network) and returns a processor instance.
  """
  def _factory(out_size: int):
    if kind == 'deepsets':
      processor = DeepSets(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    elif kind == 'gat':
      processor = GAT(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gat_full':
      processor = GATFull(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2':
      processor = GATv2(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_full':
      processor = GATv2Full(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'gatv2_ntm':
      processor = Gatv2_NTM(
          out_size=out_size,
          nb_heads=nb_heads,
          use_ln=use_ln
      )
    elif kind == 'memnet_full':
      processor = MemNetFull(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'memnet_masked':
      processor = MemNetMasked(
          vocab_size=out_size,
          sentence_size=out_size,
          linear_output_size=out_size,
      )
    elif kind == 'mpnn':
      processor = MPNN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    elif kind == 'pgn':
      processor = PGN(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    elif kind == 'pgn_mask':
      processor = PGNMask(
          out_size=out_size,
          msgs_mlp_sizes=[out_size, out_size],
          use_ln=use_ln
      )
    else:
      raise ValueError('Unexpected processor kind ' + kind)

    return processor

  return _factory
