from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ShootPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "franka_shoot"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1ShootPPORunnerCfg(ShootPPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "g1_shoot"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class G1ShootFormPPORunnerCfg(G1ShootPPORunnerCfg):
    experiment_name = "g1_shoot_form"


@configclass
class G1ShootAMPPPORunnerCfg(G1ShootPPORunnerCfg):
    experiment_name = "g1_shoot_amp"
    max_iterations = 1500


@configclass
class G1ShootDRPPORunnerCfg(G1ShootPPORunnerCfg):
    experiment_name = "g1_shoot_dr"
    max_iterations = 1000


@configclass
class G1ShootRobustPPORunnerCfg(G1ShootPPORunnerCfg):
    experiment_name = "g1_shoot_robust"
    max_iterations = 4500
    # asymmetric actor-critic: the critic reads the privileged "critic" group
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}


@configclass
class G1ShootOneHandPPORunnerCfg(G1ShootRobustPPORunnerCfg):
    experiment_name = "g1_shoot_onehand"
    max_iterations = 1500
