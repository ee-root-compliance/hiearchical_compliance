[← README](../README.md) · [简体中文](ceer1_zh.md)

# Low-level policy training

This guide covers the available low-level policies, their shared training workflow, and how to use the resulting checkpoints. All options use `train.sh`.

## Policy options

| Policy | Pipeline | Purpose | Configuration |
|---|---|---|---|
| CEER1 compliant low-level | `ceer1` | Yield to external forces while tracking motion | [G1_3kp.yaml](../cfg/task/G1/G1_3kp.yaml) |
| CEER2 stiff backbone | `stiff30` (default) | Track commands under disturbances, beneath high-level compliance policies | [G1_3kp_stiff.yaml](../cfg/task/G1/G1_3kp_stiff.yaml) |
| Fixed-stiffness baseline | `fixed_ee` | End-to-end EE compliance at 200 N/m | [G1_3kp_ee_net_pull_force_b.yaml](../cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml) |
| Ranged-stiffness baseline | `range_100_600` | End-to-end independently commanded xyz compliance over 100–600 N/m | [G1_3kp_ee_xyz_range_100_600.yaml](../cfg/task/G1/G1_3kp_ee_xyz_range_100_600.yaml) |
| 5-keypoint policy | `5kp` | Optional five-keypoint control | [G1_5kp.yaml](../cfg/task/G1/G1_5kp.yaml) |

## 1. Prepare the environment and data

Complete [installation](installation.md) and [data preparation](data.md), then configure W&B:

```bash
wandb login
export WANDB_ENTITY=your-wandb-user-or-team
export LOW_WANDB_PROJECT=my-low-level
```

## 2. Launch training

Select a `PIPELINE` from the table above. This example trains the CEER1 compliant policy:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=ceer1 bash train.sh
```

Four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 PIPELINE=ceer1 bash train.sh
```

For the CEER2 stiff backbone, replace `PIPELINE=ceer1` with `PIPELINE=stiff30`. Select other policies the same way. If omitted, `PIPELINE` defaults to `stiff30`.

The script automatically runs train → adapt → finetune, using 4B, 1B, and 2B environment frames respectively. Each stage loads the previous checkpoint through W&B. Frame counts are accumulated across environments.

## 3. Use the trained policy

The final run name depends on the selected policy. For CEER1:

```text
<entity>/<project>/ceer1_3kp_compliant_finetune_<timestamp>
```

Pass this path as `--run_path` for [interactive teleoperation](teleoperation.md) or [evaluation](evaluation.md).

After training the CEER2 stiff backbone, use the printed `LOW_RUN_PATH` for [high-level training](ceer2.md).

## How the training objectives differ

**CEER2 stiff:** keeps external disturbances but disables compliance target shifts, force-matching rewards, and excess-force penalties, learning to preserve tracking under disturbances.

**CEER1 compliant:**

Spring anchors apply external forces during training. An admittance model combines those forces with the motion reference to generate position and velocity targets that can yield under load. Rewards encourage the policy to track these targets while preserving whole-body motion and stability.

<details>
<summary>Parameters and implementation</summary>

- The command class defaults to `compliance=True` and `external_force_mode=legacy`. The legacy mode applies displacement-dependent spring forces from sampled anchors; this is not the fixed-force `net_pull` baseline.
- A numerical admittance model generates compliant position and velocity targets. Its drive pulls toward the motion reference, while external spring forces allow the target to yield.
- Training rewards track these targets, match the expected force, and penalize excessive force, alongside root tracking, EE tracking, and locomotion rewards.
- The current configuration sets `max_force=30`, `net_force_limit=30`, `force_safe_bounds=[5,15]`, and `force_safe_default=10` (N). The default force-body patterns include shoulders, wrists, and hands.

In the target model, the drive gain is `K_p = force_safe_limit / 0.05`, giving 100–300 N/m for the configured safe-limit interval. The drive force is saturated; this gain is **not a guaranteed measured EE stiffness**. The integrator uses mass 0.1 and damping 2.0, with an additional velocity-tracking drive term.

Implementation: [MotionTrackingCommand_impedance](../active_adaptation/envs/mdp/commands/motion_tracking.py) (`force_update_origin_and_target`, `force_apply`, and the force rewards), and [AdmittanceMassChain](../active_adaptation/envs/mdp/commands/admittance.py).

</details>

<details>
<summary>Launcher environment variables</summary>

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

</details>

