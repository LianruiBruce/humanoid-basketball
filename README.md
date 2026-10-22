# Humanoid Basketball: Whole-Body Shooting from Human Motion Capture + RL

<p align="center">
  <img src="media/wholebody_free_throw.gif" width="420" alt="Unitree G1 whole-body free throw">
  <img src="media/wholebody_jump_shot.gif" width="420" alt="Unitree G1 whole-body jump shot">
</p>
<p align="center"><i>Left: free throw (4.19 m). Right: jump shot (free-throw form with a human vertical jump, released near the
apex). Unitree G1, floating base, NVIDIA Isaac Lab.</i></p>

A Unitree G1 humanoid shoots a **regulation basketball** (size 7, 0.62 kg) at a **regulation NBA hoop**
(3.05 m rim) with its **whole body**. The robot balances on its own legs, its motion is tracked from **human motion
capture** (CMU subject 124), and it runs on Unitree's motor torque and speed limits. Two shots are learned from the
free-throw line: a set-shot free throw and a jump shot.

▶ Videos: [`media/wholebody_free_throw.mp4`](media/wholebody_free_throw.mp4) ·
[`media/wholebody_jump_shot.mp4`](media/wholebody_jump_shot.mp4) ·
[`media/wholebody_jump_shot_side.mp4`](media/wholebody_jump_shot_side.mp4) (side view) · earlier arm-only version:
[`media/demo.mp4`](media/demo.mp4)

Everything here is simulation. The numbers are simulated make rates, not real-robot results.

## Results

**Whole-body shots** (deterministic policy, 256 parallel envs, more than 1,000 shots each):

| Shot (4.19 m) | Make rate | Single-env video | Motion vs. the human reference |
|---|---|---|---|
| Free throw | **53%** | 7 of 8 | set shot, stays up, arms in follow-through |
| Jump shot | **62%** | 6 of 8 | feet 21 cm off the floor (human 23 cm), pelvis 0.95 m (human 0.96 m), body error 7 cm at the release |

Both start from the same human-motion tracking policy. Before the ball-flight tracking reward described below,
the make rate was about 0.2%.

**Earlier arm-only version** (fixed pelvis, Inspire hands; [Part 2](#part-2-arm-only-shooting-with-an-adversarial-motion-prior)):
84–90% from 2 m out to the 3-point line in clean simulation, and 72% / 49% (2 m / free throw) under full domain
randomization.

## Part 1: Whole-body shooting

### 1. Retargeting human mocap to the whole G1
[`scripts/retarget_cmu_wholebody.py`](scripts/retarget_cmu_wholebody.py)
- The human root trajectory is scaled by the leg-length ratio (0.736). Human segment directions are matched by
  29-DoF damped-least-squares IK in the pelvis frame, with every frame in its own sim env so all frames solve in
  parallel. Mean error is 0.3–0.9 cm.
- **Foot contact cleanup:** stance phases are detected on the human foot and the robot foot is pinned during them
  (median slide < 1 cm/s). A temporal smoothness term stops neighbouring frames from landing on different wrist IK
  branches.
- **Room for the ball:** the subject's hands sit closer together than a real ball allows, so before the release
  the wrist and knuckle targets are kept at least a ball's width apart (`--ball_hold`).

### 2. Whole-body motion tracking (no ball)
[`tracking_env_cfg.py`](source/basketball/tracking_env_cfg.py), [`tracking_mdp.py`](source/basketball/tracking_mdp.py)

A BeyondMimic-style tracker on the floating-base G1 (29 body joints and Dex3 hands):
- reference-state initialization;
- torso anchor;
- exponential-kernel rewards on body poses and velocities;
- fall terminations;
- an asymmetric critic;
- PD gains derived from motor armature.

It follows the full free-throw clip (100/100 frames).

### 3. Ball and hoop
[`shoot_track_env_cfg.py`](source/basketball/shoot_track_env_cfg.py)
- **Grasp:** until the release frame, the ball is held rigidly at the palm center, i.e. its pose and velocity are set
  to the palms'. The reaction force `m (a - g)` is applied to the palms, so the robot carries the ball's weight and
  inertia. After the release frame the ball flies with exactly the velocity the whole body gave the hands.
- **Release speed was the hard part.** The retargeted push only reaches ~2–3 m/s at the palms, because the human's
  final ball speed comes from a wrist and finger flick the IK can't reproduce. A free throw needs ≥ 7.7 m/s, and
  just reaching rim height takes 5.6 m/s vertical. PPO plateaued under a dense release-velocity reward, freed arms,
  stiffer arm gains and more exploration noise.
- **Push search** ([`scripts/cem_push.py`](scripts/cem_push.py)): the cross-entropy method (CEM) over 24 joint-offset
  parameters, applied on top of the balancing tracking policy (256 envs, ~10 min). It searches for a push that gives
  the right release velocity while the robot stays upright through the whole ball flight. Unconstrained, it found a
  full-body dive that reaches the rim and crashes, so the leg offsets are kept small and the robot must stay up for
  45 frames after the release.
- **Push + RL residual:** the push is a feedforward term in the action
  (`PushJointPositionAction`), and PPO learns the corrections on top of it.
- **Synthesized ball-flight reference** (`ball_track`): the mocap has no ball, so at the release each env gets the
  ideal 52° parabola from its own release point into the rim center. The ball is then rewarded for following that
  parabola on every frame of the flight, similar to SkillMimic's object tracking. A one-time aim score at the release
  was ~0 for nearly every shot. The dense version grades a slightly long, short or wide shot all the way to the rim,
  and it took the make rate from ~0.2% to over 50%.
- **Staying upright:** fall terminations stop shortly after the release so the shot can resolve. A dense
  post-release fall penalty keeps the robot from collapsing once it has let go.

### 4. Jump shot: free-throw form + a human vertical jump
[`scripts/compose_jump.py`](scripts/compose_jump.py)

The first jump shot followed the CMU jump-shot clip (124_05) and scored 59%, but it looked nothing like one. The
robot shot flat-footed (pelvis 15 cm below the human at the release, feet barely off the floor) with a forward chest
push (arms 30–50 cm off the human). The push search was free to trade the human form for ball speed. Adding arm and
jump costs to the search showed why: at the apex, with the arms overhead, the G1 can't produce the release speed.
The human gets it from a wrist and finger flick, and the G1 wrist only has 5 Nm.

What worked instead was keeping the free throw's upper body, which already had human-like form and a working push,
and adding a jump:
- **Motion composition:** waist, arms and pelvis orientation come from the free throw. Legs and pelvis height come
  from the jump-shot clip, shifted 7 frames so its apex lands at the free-throw release. A second pass puts the stance
  feet back on the floor.
- **Keeping the jump:** the feet must follow the reference until the release, and torso height near the release is
  rewarded. The foot-height tolerance is tightened by a curriculum from 0.13 m to 0.08 m whenever fewer than 2% of
  episodes fail it. Tightening it in one step ended every episode before the release.
- **Warm start** from the free-throw policy. The jump adds vertical speed at the release, and the ball-flight reward
  re-tunes the aim.

### Pitfalls found along the way

| Symptom | Cause | Fix |
|---|---|---|
| Ball drops 51–87% of the time, worse with a tighter grip | The Dex3 `palm_link` origin is at the heel of the hand, and the retargeted hands overlapped the ball, so depenetration pried the soft arms apart | Palm-center grasp point, ball–robot contacts filtered on every link |
| Warm-started tracker falls in 8 steps | Curled default finger pose put the finger observations far outside the normalizer's range | Fingers at the tracking defaults |
| Ball leaves at half the hand speed | Spring-damper grasp lagged the hands in a fast push | Rigid hold until the release |
| Release frame on the follow-through | It was taken from the peak of the human wrist height | Frame where the reference hands separate |
| Push found by search doesn't reproduce in the env | The policy's last-action observation didn't include the push | Last-action observation reads the action term, push included |
| PPO collapses within 6 iterations | All envs reset in lockstep, so each batch saw one motion phase | Starts spread over the first 10 frames |
| 0% with one env, 70% with 64 envs | Isaac Lab's wrench composer caches link orientations until some env resets, so with one env (or envs in lockstep) global forces were rotated by stale palm orientations | Reset the composer every step |
| Jump shot falls after every made basket | Fall terminations stop at the release window | Post-release fall penalty |
| "Jump shot" that never jumps | Searched push + freed arms + no foot check: shooting flat-footed was easier | Free-throw form + composed human jump, foot-height curriculum |
| Robot settles for an 8 cm hop | Foot check (0.18 m) loose enough to pass without jumping | Curriculum to 0.08 m, wider jump-reward kernel |

## Part 2: Arm-only shooting with an Adversarial Motion Prior

The first version of the project: G1 + Inspire hands, fixed pelvis, a distance curriculum from 2 m to the 3-point
line, and form from mocap through AMP.

<p align="center"><img src="media/free_throw.gif" width="480" alt="Arm-only free throw"></p>

**Clean simulation** (deterministic policy, ~2,000 shots per distance):

| Distance | 2 m | 3 m | Free throw 4.19 m | 5 m | 5.5 m | 6 m | 3-pointer 6.75 m |
|---|---|---|---|---|---|---|---|
| RL + form rewards | 86% | 85% | 85% | 89% | 90% | 86% | 84% |
| **+ human motion prior (AMP)** | **86%** | **89%** | **87%** | **85%** | **90%** | **90%** | **87%** |

**Robustness** (control latency 0–10 ms, sensor noise, randomized ball / link mass, friction, restitution, motor
gains and joint friction):

| Policy | 2 m | Free throw | 3-pointer |
|---|---|---|---|
| Trained without randomization | 0% | 0% | 0% |
| Fine-tuned with domain randomization | 31% | 32% | 18% |
| **From scratch with DR + 3-frame history + asymmetric actor-critic** | **72%** | **49%** | 0% |

- **Scene** ([`scripts/make_nba_hoop.py`](scripts/make_nba_hoop.py)): procedurally generated regulation hoop
  (32-capsule rim, backboard, net) and half-court markings.
- **Feasibility first** ([`scripts/feasibility_g1.py`](scripts/feasibility_g1.py)): torque-saturated swings reach
  11.8 m/s palm speed, against 7.4 m/s needed for a free throw.
- **Distance curriculum with review** (`reset_hoop_curriculum` in [`g1_mdp.py`](source/basketball/g1_mdp.py)): half
  the episodes use the frontier distance, half review learned ones (training only the frontier forgot short shots).
- **AMP** ([`amp.py`](source/basketball/amp.py)): an LSGAN discriminator over arm-feature transitions, trained inside
  the reward term. The first attempt saturated and was fixed with a smaller discriminator, input noise, a stronger
  gradient penalty, fewer updates, 100 Hz references, and a nearest-frame pose-matching term.
- **Robustness:** sub-step latency ([`DelayedJointPositionAction`](source/basketball/g1_mdp.py)), a 3-frame
  observation history for the actor, and a critic that sees the hidden randomization parameters.

Reward hacking found in this part: the policy dumped the ball when per-step posture penalties made ending the
episode cheapest; farmed a release-style bonus without making shots; held the ball until time-out to collect style
rewards; and gave up on long shots when distance shaping was ~0 for every long miss.

## Repository layout

```
source/basketball/
  tracking_env_cfg.py   whole-body motion tracking task (floating-base G1)
  tracking_mdp.py       motion command, grasp, push action, ball-flight reward, tracking rewards, terminations
  shoot_track_env_cfg.py  whole-body shooting tasks (free throw, jump shot; push + residual variants)
  g1_shoot_env_cfg.py, g1_mdp.py, amp.py   arm-only tasks (Part 2)
  mocap/cmu.py          ASF/AMC reader and forward kinematics
  assets/nba_hoop.usda  generated regulation hoop + court
scripts/
  retarget_cmu_wholebody.py, prepare_motion.py, play_wholebody_reference.py   whole-body mocap pipeline
  cem_push.py           push search (CEM) on top of a tracking policy
  compose_jump.py       jump shot reference: free-throw upper body + jump-shot legs
  style_check.py        per-frame distance to the human reference (body, pelvis and foot height)
  record_shot.py        whole-body make-rate evaluation and video recording
  debug_shoot_policy.py, survival_test.py, speed_test.py, check_ball_filter.py   diagnostics
  train.py / play.py    RSL-RL train / play (from Isaac Lab)
  retarget_cmu_to_g1.py, eval_g1.py, showcase.py, ...   arm-only pipeline (Part 2)
data/mocap/             CMU subject 124 clips, retargeted references (arm-only and whole-body)
data/motions/           tracking targets (*.npz) and searched pushes (push_*.npz)
checkpoints/            g1_wholebody_free_throw.pt, g1_wholebody_jump_shot.pt, g1_amp.pt, g1_robust.pt
```

| Task ID | What it trains |
|---|---|
| `Basketball-G1-Track-v0` | whole-body motion tracking (no ball) |
| `Basketball-G1-ShootPush-v0` | whole-body free throw: tracking + push + RL residual (`checkpoints/g1_wholebody_free_throw.pt`) |
| `Basketball-G1-ShootJumpPush-v0` | whole-body jump shot: free-throw form + composed human jump (`checkpoints/g1_wholebody_jump_shot.pt`) |
| `Basketball-G1-JumpShotPush-v0` | first jump-shot attempt on the CMU jump-shot clip (scores, but shoots flat-footed) |
| `Basketball-G1-ShootTrack-v0`, `Basketball-G1-JumpShot-v0` | tracking with ball, no push (release too slow to score) |
| `Basketball-G1-ShootAMP-v0`, `Basketball-G1-ShootRobust-v0` | arm-only (Part 2) |

Every task also has a `-Play-v0` variant.

## Setup

Tested on Windows 11, RTX 4080 16 GB, Isaac Sim 5.1 + Isaac Lab v2.3.2 (Python 3.11, torch 2.7.0+cu128).

1. Install Isaac Sim 5.1 and Isaac Lab 2.3.2 following the
   [Isaac Lab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
   (`isaaclab.bat --install rsl_rl`).
2. Known fixes on this setup:
   - `flatdict` build fails: `pip install "setuptools<81"` then `pip install flatdict==4.0.1 --no-build-isolation`.
   - `tensordict` must match torch 2.7: `pip install "tensordict>=0.8,<0.9"`.
   - Rendering (GUI or video) crashes in `rtx.scenedb.plugin` on NVIDIA drivers ≥ 595: add
     `--kit_args="--/app/vulkan=false"` (DX12). GUI mode also needs `import h5py` before the app launches.
3. Run with `OMNI_KIT_ACCEPT_EULA=YES` once you have accepted the NVIDIA Omniverse license.

## Usage

```bash
# whole-body free throw: make rate over 256 envs, then a video
python scripts/record_shot.py --headless --eval --task Basketball-G1-ShootPush-v0 --num_envs 256 --steps 600 --checkpoint checkpoints/g1_wholebody_free_throw.pt
python scripts/record_shot.py --headless --task Basketball-G1-ShootPush-v0 --steps 1000 --checkpoint checkpoints/g1_wholebody_free_throw.pt --kit_args="--/app/vulkan=false"

# whole-body jump shot, plus how close it stays to the human reference (body error, pelvis and foot height per frame)
python scripts/record_shot.py --headless --eval --task Basketball-G1-ShootJumpPush-v0 --num_envs 256 --steps 600 --checkpoint checkpoints/g1_wholebody_jump_shot.pt
python scripts/style_check.py --headless --task Basketball-G1-ShootJumpPush-Play-v0 --checkpoint checkpoints/g1_wholebody_jump_shot.pt

# whole-body pipeline from scratch
python scripts/retarget_cmu_wholebody.py --headless --amc data/mocap/cmu_124/124_04.amc --start 3.4 --end 5.4 --ball_hold --out data/mocap/wholebody/g1_124_04_ball.npz
python scripts/prepare_motion.py --headless --ref data/mocap/wholebody/g1_124_04_ball.npz --out data/motions/g1_124_04_ball.npz
python scripts/train.py --task Basketball-G1-Track-v0 --headless
python scripts/cem_push.py --headless --cost miss --leg_sigma 0.1 --upright_frames 45 --reg 1.5 --roll_sigma 0.2 --motion data/motions/g1_124_04_ball.npz --checkpoint <tracking-with-ball checkpoint> --out data/motions/push_ft.npz
python scripts/train.py --task Basketball-G1-ShootPush-v0 --headless --resume --load_run <run> --checkpoint <model>
# jump shot: compose the reference (see the script header for the 4-step compose / FK / ground-fix sequence), then
# fine-tune the free-throw policy on it
python scripts/compose_jump.py --out data/mocap/wholebody/g1_124_04_jump.npz
python scripts/train.py --task Basketball-G1-ShootJumpPush-v0 --headless --resume --load_run <free-throw run> --checkpoint <model>

# arm-only (Part 2)
python scripts/eval_g1.py --headless --task Basketball-G1-ShootRobust-v0 --checkpoint checkpoints/g1_robust.pt --distances 2.0 4.19 6.75
```

## Limitations

- **Simulation only:** no real-robot deployment. The policy gets the ball and hoop positions as state, not from a
  camera.
- **Grasp abstraction:** before the release the ball is held rigidly at the palm center and ball–robot contacts are
  filtered. The Dex3 fingers do not grip the ball.
- **The push is not from mocap:** human motion is tracked up to the set point, but the fast push phase comes from
  the CEM search plus RL. The shots read as a two-hand overhead push, not a one-hand NBA release with a guide hand.
- **The jump shot is composed:** its upper body is the free throw and its legs come from a different clip. The jump
  peaks a few frames later than the human's, so the ball leaves slightly before the apex.
- **One distance:** both shots are from the free-throw line, no shot-distance curriculum yet.
- **Fragility:** the make rate is sensitive to tiny state differences; observation noise is kept at 30% of the
  tracking level.

## Related work

- **SkillMimic** (Wang et al., CVPR 2025) and **SkillMimic-V2** (SIGGRAPH 2025) learn basketball skills (shooting,
  layups, dribbling) for physics-based humanoids from human–ball interaction data. The ball-flight tracking reward
  here follows the same idea of tracking the object, with a synthesized ball trajectory because the CMU mocap has no
  ball.
- **HumanX** (2026) learns interaction skills including a fadeaway jump shot from human videos and runs them on a
  real Unitree G1.
- **BeyondMimic** (2025): the whole-body tracking formulation used in step 2.
- Whole-body throwing with legged robots (2024) and accurate whole-body throwing with a residual policy (2025): the
  push + residual structure here is close to the latter.
- X. B. Peng et al., *AMP: Adversarial Motion Priors*, SIGGRAPH 2021 (Part 2).
- L. Liu and J. Hodgins, *Learning Basketball Dribbling Skills Using Trajectory Optimization and Deep RL*,
  SIGGRAPH 2018.

## Acknowledgements

- Motion capture: the data used in this project was obtained from [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu).
  The database was created with funding from NSF EIA-0196217.
- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause); `scripts/train.py`, `play.py`, and
  `cli_args.py` are adapted from it. Unitree G1 assets are loaded from Isaac Lab's asset server.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
