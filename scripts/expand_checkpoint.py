"""Warm-start a policy whose observations gained new terms at the END of a group.

The first linear layer of the actor / critic gets zero columns for the new inputs (so the new inputs initially
do not change the output) and the observation normalizers get mean 0 / var 1 for them. The Adam state is
cleared, since its moment buffers have the old shapes.

    python scripts/expand_checkpoint.py --src logs/rsl_rl/g1_tracking/<run>/model_500.pt \
        --dst logs/rsl_rl/g1_shoot_track/0_track_init/model_500.pt --actor_extra 10 --critic_extra 9
"""

import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True)
parser.add_argument("--dst", required=True)
parser.add_argument("--actor_extra", type=int, required=True)
parser.add_argument("--critic_extra", type=int, required=True)
args = parser.parse_args()

ck = torch.load(args.src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
for net, extra in (("actor", args.actor_extra), ("critic", args.critic_extra)):
    w = sd[f"{net}.0.weight"]
    sd[f"{net}.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra)], dim=1)
    norm = f"{net}_obs_normalizer"
    sd[f"{norm}._mean"] = torch.cat([sd[f"{norm}._mean"], torch.zeros(1, extra)], dim=1)
    sd[f"{norm}._var"] = torch.cat([sd[f"{norm}._var"], torch.ones(1, extra)], dim=1)
    sd[f"{norm}._std"] = torch.cat([sd[f"{norm}._std"], torch.ones(1, extra)], dim=1)
    print(f"{net}: inputs {w.shape[1]} -> {sd[f'{net}.0.weight'].shape[1]}")
ck["optimizer_state_dict"]["state"] = {}
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
torch.save(ck, args.dst)
print(f"saved {args.dst} (iter {ck['iter']})")
