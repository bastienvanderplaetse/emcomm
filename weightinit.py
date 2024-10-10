import torch

def custom_init(m):
    depth = m.shape[0]
    side = m.shape[1] // depth

    for l in range(depth):
        min_idx = side * l
        max_idx = side * (l+1)
        mask = torch.ones_like(m[l], dtype=torch.bool)
        mask[min_idx:max_idx] = False
        m[l][mask] = 0.