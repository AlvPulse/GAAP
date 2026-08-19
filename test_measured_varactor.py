from beamformer.element_model import MeasuredVaractor
from compute_constants import compute_card

print("MeasuredVaractor (Folding=False)")
v1 = MeasuredVaractor(folding=False)
c1 = compute_card(v1.c_grid)
for k, v in c1.items():
    print(f"  {k}: {v}")

print("\nMeasuredVaractor (Folding=True)")
v2 = MeasuredVaractor(folding=True)
c2 = compute_card(v2.c_grid)
for k, v in c2.items():
    print(f"  {k}: {v}")
