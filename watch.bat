@echo off
rem Open Isaac Sim and watch the human-like (AMP) shooting policy.
rem Run from an environment where `python` is the Isaac Lab Python (e.g. the conda env with isaacsim installed).
rem DX12 renderer: Isaac Sim 5.1's Vulkan RTX path crashes on NVIDIA drivers >= 595.
set OMNI_KIT_ACCEPT_EULA=YES
cd /d "%~dp0"
python scripts\play.py --task Basketball-G1-ShootAMP-Play-v0 --checkpoint checkpoints\g1_amp.pt --kit_args="--/app/vulkan=false" %*
