"""Quick diagnostic for the captured Huginn activations.

Loads runs/svamp_huginn/activations.pt and prints two grids:
  1. mean |activation| per (step, layer)  — zero rows imply broken capture
  2. std across questions per (step, layer) — zero implies constant tensor
Run from the inference/ directory.
"""

import torch

acts = torch.load(
    "runs/svamp_huginn/activations.pt", map_location="cpu", weights_only=False
)
print(f"shape={tuple(acts.shape)} dtype={acts.dtype}")
N, S, L, H = acts.shape


def grid(title, fn):
    print()
    print(title)
    header_cells = ["L%d" % l for l in range(L)]
    print("step " + "".join(f"{c:>10}" for c in header_cells))
    for s in range(S):
        row_cells = [f"{fn(s, l):>10.4f}" for l in range(L)]
        print(f"{s:>4} " + "".join(row_cells))


grid(
    "Mean |activation|  (zero -> broken capture or dead path)",
    lambda s, l: acts[:, s, l, :].float().abs().mean().item(),
)

grid(
    "Std across N samples (zero -> constant tensor)",
    lambda s, l: acts[:, s, l, :].float().std().item(),
)
