
class FixLenIter(object):
  def __init__(self, iterator, length):
    self.iter = iterator
    self.length = length
    self.count = 0
  def __next__(self):
    if self.count < self.length:
      self.count += 1
      return next(self.iter)
    else:
      self.count = 0
      raise StopIteration()
  def __iter__(self):
    return self
    