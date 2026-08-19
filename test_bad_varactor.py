from beamformer.element_model import SevereVDILVaractor
from compute_constants import compute_card

print("SevereVDILVaractor (Folding=False)")
v1 = SevereVDILVaractor(folding=False)
c1 = compute_card(v1.c_grid)
for k, v in c1.items():
    print(f"  {k}: {v}")

print("\nSevereVDILVaractor (Folding=True)")
v2 = SevereVDILVaractor(folding=True)
c2 = compute_card(v2.c_grid)
for k, v in c2.items():
    print(f"  {k}: {v}")
