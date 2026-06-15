import os

filepath = 'beamformer/iterative.py'
with open(filepath, 'r') as f:
    content = f.read()

# Replace any lingering 2-tuple unpacking with 3-tuple
content = content.replace('V_n[n], c_n[n] = element.project(rotated_cand[n]', 'V_n[n], c_n[n], _ = element.project(rotated_cand[n]')
content = content.replace('V_cand[n], c_cand[n] = element.project(', 'V_cand[n], c_cand[n], _ = element.project(')

with open(filepath, 'w') as f:
    f.write(content)
print("done")
