# Humanoid Basketball Shooting: RL + Human Motion Prior

<p align="center"><img src="media/free_throw.gif" width="560" alt="Unitree G1 making a free throw in Isaac Lab"></p>

A Unitree G1 humanoid with dexterous hands learns to shoot a **regulation basketball** (size 7, 0.62 kg) at a
**regulation NBA hoop** (3.05 m rim, free-throw and 3-point lines) in NVIDIA Isaac Lab. It does this with
**Unitree's real arm motor limits** (25 Nm, 37 rad/s), and its shooting form comes from **human motion capture**
through an Adversarial Motion Prior.

▶ **30-second demo:** [`media/demo.mp4`](media/demo.mp4)

Everything here is simulation. The numbers are simulated make rates, not real-robot results.

## Results

**Clean simulation** (deterministic policy, ~2,000 shots per distance):

| Distance | 2 m | 3 m | Free throw 4.19 m | 5 m | 5.5 m | 6 m | 3-pointer 6.75 m |
|---|---|---|---|---|---|---|---|
| RL + form rewards | 86% | 85% | 85% | 89% | 90% | 86% | 84% |
| **+ human motion prior (AMP)** | **86%** | **89%** | **87%** | **85%** | **90%** | **90%** | **87%** |

The motion prior made the motion human-like at no cost in accuracy. The discriminator's score on policy motion
reached −0.40 (+1 = indistinguishable from the human; the first, saturated AMP attempt was stuck at −0.98), and
pose similarity to the reference reached 0.94.

**Robustness**: tested with control latency (0–10 ms), sensor noise, and randomized ball / link mass, friction,
restitution, motor gains and joint friction:

| Policy | 2 m | Free throw | 3-pointer |
|---|---|---|---|
| Trained without randomization | 0% | 0% | 0% |
| Fine-tuned with domain randomization | 31% | 32% | 18% |
| **Trained from scratch with DR + 3-frame observation history + asymmetric actor-critic** | **72%** | **49%** | 0%* |

The policy trained without randomization drops the ball once 5–10 ms of latency is added: it had overfit to
perfectly timed control. Domain randomization brought ball drops from ~60% to ~0.1%.
\*The from-scratch robust policy never learned enough power for the 3-point range under full randomization (see
[Limitations](#limitations)).

## How it works

### 1. Task and physics
- **Scene** ([`scripts/make_nba_hoop.py`](scripts/make_nba_hoop.py)): a procedurally generated regulation hoop
  (32-capsule rim, backboard, net, support) and half-court markings.
- **Robot** ([`g1_shoot_env_cfg.py`](source/basketball/g1_shoot_env_cfg.py)): G1 + Inspire hands, fixed pelvis,
  gravity on, arm actuators clipped to Unitree's URDF limits. The ball starts squeezed between the palms, and there is
  no "release" action: the ball leaves the hands only through arm motion.
- **Feasibility first** ([`scripts/feasibility_g1.py`](scripts/feasibility_g1.py)): before training, torque-saturated
  swings showed the arms reach 11.8 m/s palm speed, against 7.4 m/s needed for a free throw and 8.9 m/s for a
  3-pointer.

### 2. Learning to shoot (PPO, 4,096 parallel envs, one RTX 4080)
- Rewards: made shot, a dense "ball near the hoop" term, a drop penalty, and form terms (release height and launch
  angle paid only on makes, entry angle, ball kept on the shooting line, level hands).
- **Distance curriculum with review** ([`g1_mdp.py`](source/basketball/g1_mdp.py) `reset_hoop_curriculum`): half the
  episodes use the current farthest distance, half re-sample distances already learned. The first version only
  trained the frontier and forgot short shots (2 m make rate fell to 2%).

### 3. Human-like form from motion capture
- **Data:** CMU Graphics Lab Motion Capture Database, subject 124 (basketball shot, free throw, jump shot).
- **Retargeting** ([`scripts/retarget_cmu_to_g1.py`](scripts/retarget_cmu_to_g1.py)): upper-arm, forearm and hand
  directions are taken in the human torso frame and rebuilt with the G1's segment lengths. Damped-least-squares IK is
  then solved for both 7-DoF arms, with every frame in its own sim env so all frames solve in parallel (0.6–0.7 cm
  mean error).
- **AMP** ([`source/basketball/amp.py`](source/basketball/amp.py)): an LSGAN discriminator over (s, s′) arm-feature
  transitions, trained inside the reward term so RSL-RL is unmodified. The first attempt saturated: the discriminator
  scored every policy transition at −1 and the style reward went flat. It was fixed with a smaller discriminator,
  input noise, a stronger gradient penalty, fewer updates, 100 Hz references, and a time-free nearest-frame
  pose-matching term.

### 4. Robustness (sim-to-real style)
- Domain randomization of ball mass (567–650 g, the NBA spec), friction and restitution, link mass, motor gains,
  joint friction, palm friction, 0–10 ms sub-step control latency
  ([`DelayedJointPositionAction`](source/basketball/g1_mdp.py)), and sensor noise.
- The **actor** sees the last 3 noisy observations, so it can infer latency and motor strength. The **critic** sees
  the noise-free state plus the hidden randomization parameters ([`privileged_dynamics`](source/basketball/g1_mdp.py)),
  i.e. an asymmetric actor-critic.

### Reward hacking found and fixed

| Symptom | Cause | Fix |
|---|---|---|
| Policy dumps the ball immediately (4-step episodes) | Per-step posture penalties made ending the episode the cheapest option | Drop termination with a penalty; posture penalties switched on later by curriculum |
| Pretty releases that never go in | Release-style bonus was easier to farm than making shots | Style bonus held until the shot is made |
| Holds the ball until time-out | Per-step style rewards (AMP, pose match) pay for the holding pose too | Style rewards enabled only after the policy can shoot |
| Gives up on long shots (pushes the ball 1 m) | Distance-based shaping is ~0 for every long miss | Ballistic aim reward at release, scaled with shot distance (work in progress) |

## Repository layout

```
source/basketball/
  g1_shoot_env_cfg.py   G1 tasks: base, form rewards, AMP, domain randomization, robust (history + asym. critic)
  g1_mdp.py             actions (latency), observations, rewards, terminations, curriculum
  amp.py                Adversarial Motion Prior + pose-match rewards
  mocap/cmu.py          ASF/AMC reader and forward kinematics
  assets/nba_hoop.usda  generated regulation hoop + court
  shoot_env_cfg.py, mdp.py   early Franka-arm prototype
scripts/
  train.py / play.py    RSL-RL train / play (from Isaac Lab, registers these tasks)
  eval_g1.py            hit rate per distance (--clean disables randomization)
  showcase.py           record a fixed-distance shot video
  retarget_cmu_to_g1.py, view_mocap.py, play_reference.py   mocap pipeline
  feasibility_g1.py, hold_test_g1.py, debug_dr.py            physics checks and DR ablations
  make_nba_hoop.py, make_demo.py
data/mocap/             CMU subject 124 clips + retargeted G1 references (.npz)
checkpoints/            g1_amp.pt (clean-sim best), g1_robust.pt (best under randomization)
```

| Task ID | What it trains |
|---|---|
| `Basketball-G1-ShootForm-v0` | PPO + form rewards + distance curriculum |
| `Basketball-G1-ShootAMP-v0` | + Adversarial Motion Prior (use `checkpoints/g1_amp.pt`) |
| `Basketball-G1-ShootDR-v0` | + domain randomization |
| `Basketball-G1-ShootRobust-v0` | DR + observation history + asymmetric critic (use `checkpoints/g1_robust.pt`) |

## Setup

Tested on Windows 11, RTX 4080 16 GB, Isaac Sim 5.1 + Isaac Lab v2.3.2 (Python 3.11, torch 2.7.0+cu128).

1. Install Isaac Sim 5.1 and Isaac Lab 2.3.2 following the
   [Isaac Lab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
   (`isaaclab.bat --install rsl_rl`).
2. Known fixes on this setup:
   - `flatdict` build fails: `pip install "setuptools<81"` then `pip install flatdict==4.0.1 --no-build-isolation`.
   - `tensordict` must match torch 2.7: `pip install "tensordict>=0.8,<0.9"`.
   - Rendering (GUI or `--video`) crashes in `rtx.scenedb.plugin` on NVIDIA drivers ≥ 595: add
     `--kit_args="--/app/vulkan=false"` (DX12). GUI mode also needs `import h5py` before the app launches (done in
     `scripts/play.py`).
3. Run with `OMNI_KIT_ACCEPT_EULA=YES` once you have accepted the NVIDIA Omniverse license.

## Usage

```bash
# watch the human-like policy in Isaac Sim (Windows)
watch.bat

# train
python scripts/train.py --task Basketball-G1-ShootForm-v0 --headless

# evaluate hit rate per distance (add --clean to turn off randomization)
python scripts/eval_g1.py --headless --task Basketball-G1-ShootRobust-v0 --checkpoint checkpoints/g1_robust.pt --distances 2.0 4.19 6.75

# record a 3-pointer from the side
python scripts/showcase.py --headless --task Basketball-G1-ShootAMP-Play-v0 --checkpoint checkpoints/g1_amp.pt --distance 6.75 --side --kit_args="--/app/vulkan=false"

# mocap pipeline
python scripts/retarget_cmu_to_g1.py --headless --amc data/mocap/cmu_124/124_04.amc --start 3.7 --end 5.2 --fps 100 --out data/mocap/g1_124_04_100hz.npz
python scripts/play_reference.py --headless --ref data/mocap/g1_124_04_100hz.npz --kit_args="--/app/vulkan=false"
```

## Limitations

- **Simulation only**: no real-robot deployment. The pelvis is fixed (no legs or balance), and the policy gets the
  hoop position as a state input rather than from a camera.
- **Range under randomization**: the robust policy trained from scratch stalled at 5.25 m and scores 0% from the
  3-point line under full randomization, while a policy first trained in clean sim and then fine-tuned with DR keeps
  18%. A distance-scaled ballistic aim reward is in progress; training clean-first-then-DR with the robust
  architecture is the next thing to try.
- The shot is a right-hand-dominant two-hand shot, not yet a strict one-hand shot with a passive guide hand
  (references with a frozen guide arm are already generated: `retarget_cmu_to_g1.py --one_hand`).

**Next:** strict one-hand form, whole-body jump shot (floating base + balance), head-camera perception.

## Acknowledgements

- Motion capture: the data used in this project was obtained from [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu).
  The database was created with funding from NSF EIA-0196217.
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause); `scripts/train.py`, `play.py`, and
  `cli_args.py` are adapted from it. Unitree G1 + Inspire hand assets are loaded from Isaac Lab's asset server.
- X. B. Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*, SIGGRAPH 2021.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
