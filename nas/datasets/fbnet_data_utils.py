import torchvision.datasets as datasets

import numpy as np
import pickle

class FBNet_ds(datasets.ImageFolder):
  """Image net ds folder for fbnet.
  """
  def __init__(self,
               **kwargs):
    super(FBNet_ds, self).__init__(**kwargs)
  
  def filter(self,
             samples_classes=100,
             random_seed=None,
             restore=False):
    """Get \a samples_classes from total ds.
    """
    if restore:
      try:
        with open('./tmp/classes_%d.pkl' % samples_classes, 'rb') as f:
          _classes = pickle.load(f)
        if samples_classes == len(_classes):
          with open('./tmp/class_to_idx_%d.pkl' % samples_classes, 'rb') as f:
            _class_to_idx = pickle.load(f)
          with open('./tmp/samples_%d.pkl' % samples_classes, 'rb') as f:
            _samples = pickle.load(f)
          self.classes = _classes
          self.class_to_idx = _class_to_idx
          self.samples = _samples
          return
      except Exception, e:
        print(e)
        pass
    _num_classes =  len(self.classes)
    if not random_seed is None:
      assert isinstance(random_seed, int)
      np.random.seed(random_seed)
    choosen_cls_idx = list(np.random.choice(list(range(_num_classes)), 
                                            samples_classes))
    _class_to_idx = {}
    cls_id = 0
    _cls_map = dict()
    for k, v in self.class_to_idx.items():
      if v in choosen_cls_idx:
        _class_to_idx[k] = cls_id
        _cls_map[v] = cls_id
        cls_id += 1
    if cls_id < samples_classes:
      # missing_num = samples_classes - cls_id
      for k, v in self.class_to_idx.items():
        if v not in choosen_cls_idx:
          _class_to_idx[k] = cls_id
          _cls_map[v] = cls_id
          cls_id += 1
        if cls_id == samples_classes:
          break
    assert len(_class_to_idx.keys()) == samples_classes, \
        "%d vs %d" % (len(_class_to_idx.keys()), samples_classes)
    self.class_to_idx = _class_to_idx
    with open('./tmp/class_to_idx_%d.pkl' % samples_classes, 'wb') as f:
      pickle.dump(self.class_to_idx, f)

    _samples = []
    for item in self.samples:
      if item[1] in choosen_cls_idx:
        _samples.append((item[0], _cls_map[item[1]]))
    self.samples = _samples
    with open('./tmp/samples_%d.pkl' % samples_classes, 'wb') as f:
      pickle.dump(self.samples, f)

    self.classes = list(_class_to_idx.keys())
    with open('./tmp/classes_%d.pkl' % samples_classes, 'wb') as f:
      pickle.dump(self.classes, f)