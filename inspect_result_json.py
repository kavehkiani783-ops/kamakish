import json
import glob
import os
from pprint import pprint

files = sorted(glob.glob("results/tmr_v2_listops_synth_seed*_steps*_slots*_topk*_gate1.json"))

if not files:
    print("No matching files found.")
    raise SystemExit(1)

target = files[0]
print("Inspecting:", target)

with open(target, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTop-level type:", type(data))
print("\nTop-level keys:")
if isinstance(data, dict):
    print(list(data.keys()))
else:
    print("Not a dict")

print("\nFull JSON content:")
pprint(data, sort_dicts=False)
