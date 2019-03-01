
def tensor2list(t):
  """Transfer torch.tensor to list.
  """
  return list(t.detach().long().cpu().numpy())

def scalar2int(t):
  """Transfer scalar to int.
  """
  return int(t.detach().long().cpu().numpy())
