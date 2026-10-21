"""Robot-arm basketball tasks for Isaac Lab."""

import gymnasium as gym

gym.register(
    id="Basketball-Shoot-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shoot_env_cfg:ShootEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ShootPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-Shoot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-Shoot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootForm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootFormEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootFormPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootForm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootFormEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootFormPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootAMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootAMPPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootAMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootAMPEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootAMPPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootDRPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootDR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootDREnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootDRPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootRobust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootRobustPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootRobust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootRobustEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootRobustPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootRobustFullRange-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootRobustFullRangeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootRobustPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootOneHand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootOneHandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootOneHandPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-ShootOneHand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_shoot_env_cfg:G1ShootOneHandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1ShootOneHandPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-Track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1TrackingPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-G1-Track-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1TrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:G1TrackingPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Basketball-Shoot-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shoot_env_cfg:ShootEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ShootPPORunnerCfg",
    },
    disable_env_checker=True,
)
