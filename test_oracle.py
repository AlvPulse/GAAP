from experiments.Ablation_total import run_one
from beamformer.pattern_projection import Task

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
cfg = {"desc": "test", "expected": "", "kwargs": {}, "disable_lcmv_dual": True}
res = run_one("A4_no_lcmv", cfg, task, 32, 0)
print(f"Oracle: {res.oracle}, GLCP Moves: {res.glcp_moves}")
