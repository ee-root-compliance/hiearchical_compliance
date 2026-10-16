[← README](../README.md)

## CEER1 compliance training

The compliant low-level configuration is [G1_3kp.yaml](../cfg/task/G1/G1_3kp.yaml).
Select it explicitly; the launcher's default `stiff30` trains the stiff backbone used by CEER2.

```bash
export WANDB_ENTITY=your-wandb-user-or-team
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=ceer1 \
  LOW_WANDB_PROJECT=my-low-level bash train.sh
```

This runs train → adapt → finetune with `task=G1/G1_3kp`.

### How compliance is learned

- The command class defaults to `compliance=True` and `external_force_mode=legacy`. The legacy mode applies displacement-dependent spring forces from sampled anchors; this is not the fixed-force `net_pull` baseline.
- A numerical admittance model generates compliant position and velocity targets. Its drive pulls toward the motion reference, while external spring forces allow the target to yield.
- Training rewards track these targets, match the expected force, and penalize excessive force, alongside root tracking, EE tracking, and locomotion rewards.
- The current configuration sets `max_force=30`, `net_force_limit=30`, `force_safe_bounds=[5,15]`, and `force_safe_default=10` (N). The default force-body patterns include shoulders, wrists, and hands.

In the target model, the drive gain is `K_p = force_safe_limit / 0.05`, giving 100–300 N/m for the configured safe-limit interval. The drive force is saturated; this gain is **not a guaranteed measured EE stiffness**. The integrator uses mass 0.1 and damping 2.0, with an additional velocity-tracking drive term.

Implementation: [MotionTrackingCommand_impedance](../active_adaptation/envs/mdp/commands/motion_tracking.py) (`force_update_origin_and_target`, `force_apply`, and the force rewards), and [AdmittanceMassChain](../active_adaptation/envs/mdp/commands/admittance.py).

### Choose the appropriate configuration

| Pipeline | Configuration | Behavior |
|---|---|---|
| `ceer1` | [G1_3kp.yaml](../cfg/task/G1/G1_3kp.yaml) | Compliant low-level control using anchor forces and admittance targets |
| `stiff30` (default) | [G1_3kp_stiff.yaml](../cfg/task/G1/G1_3kp_stiff.yaml) | CEER2 backbone: `compliance=False`; force matching and excess-force rewards disabled |
| `fixed_ee` | [G1_3kp_ee_net_pull_force_b.yaml](../cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml) | Fixed 200 N/m EE baseline using hand-only net-pull forces |

These descriptions reflect the configurations currently shipped in this repository. Exact reproduction of a historical CEER1 checkpoint also requires its saved training configuration.

## Low-level training

`train.sh` runs the low-level teacher-student pipeline:

```text
train       4B environment frames
adapt       1B environment frames
finetune    2B environment frames
```

All settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `WANDB_ENTITY` | (required) | W&B user or team |
| `LOW_WANDB_PROJECT` | `your-low-level-project` | W&B project for low-level runs |
| `PIPELINE` | `stiff30` | One of `ceer1`, `stiff30`, `fixed_ee`, `range_100_600`, `5kp` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPUs to use |
| `NPROC` | `4` | Processes per node; must match the number of GPUs |
| `MASTER_PORT` | `29501` | `torchrun` port |
| `RUN_TIMESTAMP` | current time | Suffix of the W&B run ids |

```bash
# Shared 3-keypoint stiff backbone; maximum external force is 30 N.
PIPELINE=stiff30 LOW_WANDB_PROJECT=my-low-level bash train.sh

# End-to-end fixed 200xyz compliance baseline.
PIPELINE=fixed_ee LOW_WANDB_PROJECT=my-low-level bash train.sh

# End-to-end independently ranged xyz compliance baseline over 100--600 N/m.
PIPELINE=range_100_600 LOW_WANDB_PROJECT=my-low-level bash train.sh

# Optional 5-keypoint locomotion policy.
PIPELINE=5kp LOW_WANDB_PROJECT=my-low-level bash train.sh

# Single GPU:
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=stiff30 LOW_WANDB_PROJECT=my-low-level bash train.sh
```

The corresponding task configs are:

```text
cfg/task/G1/G1_3kp_stiff.yaml
cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml
cfg/task/G1/G1_3kp_ee_xyz_range_100_600.yaml
cfg/task/G1/G1_5kp.yaml
```

When the pipeline finishes, it prints the run path for high-level training:

```text
LOW_RUN_PATH=<entity>/<project>/compliance_3kp_stiff_finetune_limmt_full_force30_<timestamp>
```

To start from a local file instead, pass `checkpoint_path=/path/to/checkpoint.pt`
to `scripts/train.py`.

