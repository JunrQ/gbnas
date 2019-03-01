def get_same_padding(s):
  """Given stride return pad.
  """
  assert s in [1, 3, 5, 7], "weird stride provided: %d" % s
  return int((s - 1) / 2)