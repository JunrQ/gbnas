
import torch.nn as nn

from .weigthed_sum_blks import WeightedSum

class FBNetPaperBlock(WeightedSum):
  """[FBNet](https://arxiv.org/pdf/1812.03443.pdf)
  """
  def __init__(self, **kwargs):
    """Custom blocks like FBNet.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    name : str
    device : str or torch.device
    """
    super(FBNetCustomBlock, self).__init__(**kwargs)
    self._default_cfg = None

    self.build_from_config(self.default_config())
    self.init_arch_params()

  def default_config(self):
    if self._default_cfg is None:
      cfg = []

      _e = [1, 1, 3, 6,
            1, 1, 3, 6]
      _kernel = [3, 3, 3, 3,
                 5, 5, 5, 5]
      _group = [1, 2, 1, 1,
                1, 2, 1, 1]

      for b_idx, (e, k, g) in enumerate(zip(_e, _kernel, _group)):
        kwargs = {'in_channels' : self.in_channels,
                  'out_channels' : self.out_channels,
                  'kernel_size' : k,
                  'stride' : self.stride,
                  'groups' : g,
                  'expansion' : e}
        cfg.append(['FBNetBlock', kwargs])

      self._default_cfg = cfg
    return self._default_cfg
