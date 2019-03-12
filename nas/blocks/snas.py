



from .dag_blks import DAGBlock

class SNAS(DAGBlock):



  def __init__(self, **kwargs):
    

    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr))