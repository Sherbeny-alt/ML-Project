"""Export model.joblib to docs/model.json so it can run in the browser (no server).

Usage:  python export_static.py
"""
import json
import re
import joblib
import numpy as np

model = joblib.load("model.joblib")
prep, clf = model.named_steps["prep"], model.named_steps["clf"]
booster = clf.get_booster()

scaler = prep.named_transformers_["num"].named_steps["scale"]

# --- trees as compact nested arrays: leaf = number, node = [feature, threshold, yes, no]
# Use the exact saved model (float32 thresholds); get_dump() rounds thresholds as text.
raw = json.loads(bytes(booster.save_raw(raw_format="json")))
tree_list = raw["learner"]["gradient_booster"]["model"]["trees"]

def convert_tree(t):
    left, right = t["left_children"], t["right_children"]
    feat, cond = t["split_indices"], t["split_conditions"]
    def build(i):
        if left[i] == -1:                 # leaf: value is stored in split_conditions
            return cond[i]
        return [feat[i], cond[i], build(left[i]), build(right[i])]
    return build(0)

trees = [convert_tree(t) for t in tree_list]

# --- base margin = logit(base_score), read straight from the model (binary:logistic)
base_score = float(re.findall(r"[-+0-9.eE]+", raw["learner"]["learner_model_param"]["base_score"])[0])
base = float(np.log(base_score / (1 - base_score)))

out = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "base_margin": base,
    "types": ["H", "L", "M"],          # one-hot order, features 5,6,7
    "trees": trees,
}
with open("docs/model.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"Saved docs/model.json  ({len(trees)} trees)")
