from __future__ import print_function

import torch


### ENVIRONMENT ###
verbose = True

cuda = torch.cuda.is_available()

DEVICE = "cuda" if cuda else "cpu"

v_print = print if verbose else lambda *a, **k: None


v_print(DEVICE)

tensor = torch.Tensor().to(device=DEVICE)
print(tensor)
print(tensor.new_ones(10))