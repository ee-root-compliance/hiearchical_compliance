[← 中文首页](../README_zh.md) · [English](ceer1.md)

# 底层策略训练

本文介绍底层策略的可选配置、统一训练流程和训练结果的使用方式。所有选项均通过 `train.sh` 启动。

## 策略选项

| 策略 | Pipeline | 用途 | 配置 |
|---|---|---|---|
| CEER1 柔顺底层 | `ceer1` | 在跟踪动作时对外力产生让步 | [G1_3kp.yaml](../cfg/task/G1/G1_3kp.yaml) |
| CEER2 stiff 底层 | `stiff30`（默认） | 抵抗扰动、跟踪指令，供高层柔顺策略使用 | [G1_3kp_stiff.yaml](../cfg/task/G1/G1_3kp_stiff.yaml) |
| 固定刚度 baseline | `fixed_ee` | End-to-end 固定 200 N/m EE 柔顺 | [G1_3kp_ee_net_pull_force_b.yaml](../cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml) |
| 范围刚度 baseline | `range_100_600` | End-to-end 三轴独立 100–600 N/m EE 柔顺 | [G1_3kp_ee_xyz_range_100_600.yaml](../cfg/task/G1/G1_3kp_ee_xyz_range_100_600.yaml) |
| 5-keypoint 策略 | `5kp` | 可选的五关键点控制 | [G1_5kp.yaml](../cfg/task/G1/G1_5kp.yaml) |

## 1. 准备环境与数据

先完成[环境安装](installation_zh.md)和[数据准备](data_zh.md)，再配置 W&B：

```bash
wandb login
export WANDB_ENTITY=your-wandb-user-or-team
export LOW_WANDB_PROJECT=my-low-level
```

## 2. 启动训练

先选择上表中的 `PIPELINE`。以下以 CEER1 柔顺底层为例：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=ceer1 bash train.sh
```

四张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 PIPELINE=ceer1 bash train.sh
```

训练 CEER2 stiff 底层时，将 `PIPELINE=ceer1` 替换为 `PIPELINE=stiff30`；其他选项同理。未设置 `PIPELINE` 时默认使用 `stiff30`。

脚本自动依次执行 train → adapt → finetune，分别采样 40 亿、10 亿和 20 亿环境帧，并通过 W&B 加载上一阶段的 checkpoint。这里的帧数是所有环境累计的采样量。

## 3. 使用训练结果

最终 run 名称取决于所选策略。CEER1 示例：

```text
<entity>/<project>/ceer1_3kp_compliant_finetune_<timestamp>
```

将该路径作为 `--run_path`，即可进行[交互式遥操作](teleoperation_zh.md)或[评测](evaluation_zh.md)。

训练 CEER2 stiff 底层后，使用脚本打印的 `LOW_RUN_PATH` 启动[高层训练](ceer2_zh.md)。

## 训练目标的区别

**CEER2 stiff：** 保留外力扰动，关闭柔顺目标偏移、力匹配奖励和超力惩罚，学习在扰动下保持跟踪。

**CEER1 柔顺：**

训练时，弹簧 anchor 向机器人施加外力。导纳模型结合外力和运动参考，生成允许让步的位置与速度目标；策略通过奖励学习跟踪这些目标，同时保持全身运动与稳定性。

<details>
<summary>参数与实现细节</summary>

- Command 类默认启用 `compliance=True`，外力模式为 `external_force_mode=legacy`。该模式采样弹簧 anchor，外力随机器人与 anchor 的相对位移变化，与固定外力的 `net_pull` baseline 不同。
- 数值导纳模型生成柔顺的位置与速度目标。模型中的驱动力拉向动作参考，外部弹簧力则使目标产生让步。
- 策略通过柔顺目标跟踪、期望力匹配和超力惩罚学习这种响应，同时保留 root、EE 跟踪和行走奖励。
- 当前配置为 `max_force=30`、`net_force_limit=30`、`force_safe_bounds=[5,15]`、`force_safe_default=10`（N）。默认施力 body pattern 包括肩、腕和手。

目标模型的驱动增益为 `K_p = force_safe_limit / 0.05`，对应当前安全力区间的 100–300 N/m。驱动力有饱和限制，因此该增益**不能直接视为实测 EE stiffness**。积分器使用 mass 0.1、damping 2.0，驱动力另有速度跟踪项。

实现位于 [MotionTrackingCommand_impedance](../active_adaptation/envs/mdp/commands/motion_tracking.py) 的 `force_update_origin_and_target`、`force_apply` 和 force reward 函数，以及 [AdmittanceMassChain](../active_adaptation/envs/mdp/commands/admittance.py)。

</details>

<details>
<summary>训练脚本环境变量</summary>

所有设置都可以通过环境变量传递：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `WANDB_ENTITY` | 必填 | W&B user 或 team |
| `LOW_WANDB_PROJECT` | `your-low-level-project` | 低层 W&B project |
| `PIPELINE` | `stiff30` | `ceer1`、`stiff30`、`fixed_ee`、`range_100_600` 或 `5kp` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 使用的 GPU |
| `NPROC` | `4` | 每个节点的进程数，必须与 GPU 数量一致 |
| `MASTER_PORT` | `29501` | `torchrun` 端口 |
| `RUN_TIMESTAMP` | 当前时间 | W&B run id 后缀 |

</details>

