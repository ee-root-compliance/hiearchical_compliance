[← 中文首页](../README_zh.md)

## CEER1 柔顺训练

底层柔顺配置保存在 [G1_3kp.yaml](../cfg/task/G1/G1_3kp.yaml)。需要显式选择该配置；脚本默认的 `stiff30` 训练的是 CEER2 使用的 stiff 底层。

```bash
export WANDB_ENTITY=your-wandb-user-or-team
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=ceer1 \
  LOW_WANDB_PROJECT=my-low-level bash train.sh
```

该命令使用 `task=G1/G1_3kp` 依次执行 train → adapt → finetune。

### 柔顺性如何训练

- Command 类默认启用 `compliance=True`，外力模式为 `external_force_mode=legacy`。该模式采样弹簧 anchor，外力随机器人与 anchor 的相对位移变化，与固定外力的 `net_pull` baseline 不同。
- 数值导纳模型生成柔顺的位置与速度目标。模型中的驱动力拉向动作参考，外部弹簧力则使目标产生让步。
- 策略通过柔顺目标跟踪、期望力匹配和超力惩罚学习这种响应，同时保留 root、EE 跟踪和行走奖励。
- 当前配置为 `max_force=30`、`net_force_limit=30`、`force_safe_bounds=[5,15]`、`force_safe_default=10`（N）。默认施力 body pattern 包括肩、腕和手。

目标模型的驱动增益为 `K_p = force_safe_limit / 0.05`，对应当前安全力区间的 100–300 N/m。驱动力有饱和限制，因此该增益**不能直接视为实测 EE stiffness**。积分器使用 mass 0.1、damping 2.0，驱动力另有速度跟踪项。

实现位于 [MotionTrackingCommand_impedance](../active_adaptation/envs/mdp/commands/motion_tracking.py) 的 `force_update_origin_and_target`、`force_apply` 和 force reward 函数，以及 [AdmittanceMassChain](../active_adaptation/envs/mdp/commands/admittance.py)。

### 配置区别

| Pipeline | 配置 | 训练行为 |
|---|---|---|
| `ceer1` | [G1_3kp.yaml](../cfg/task/G1/G1_3kp.yaml) | 使用 anchor 弹簧外力和导纳目标训练底层柔顺控制 |
| `stiff30`（默认） | [G1_3kp_stiff.yaml](../cfg/task/G1/G1_3kp_stiff.yaml) | CEER2 底层：关闭 `compliance`、力匹配奖励和超力惩罚 |
| `fixed_ee` | [G1_3kp_ee_net_pull_force_b.yaml](../cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml) | 仅向手部施加 net-pull 外力的固定 200 N/m EE baseline |

以上说明基于仓库当前配置。若需精确复现历史 CEER1 checkpoint，还需核对该 checkpoint 保存的训练配置。

## 低层训练

`train.sh` 执行低层 teacher-student 流程：

```text
train       4B environment frames
adapt       1B environment frames
finetune    2B environment frames
```

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

示例：

```bash
# 共享的 3-keypoint stiff backbone，最大外力为 30 N。
PIPELINE=stiff30 LOW_WANDB_PROJECT=my-low-level bash train.sh

# End-to-end 固定 200xyz compliance baseline。
PIPELINE=fixed_ee LOW_WANDB_PROJECT=my-low-level bash train.sh

# 100--600 N/m 的 end-to-end xyz ranged compliance baseline。
PIPELINE=range_100_600 LOW_WANDB_PROJECT=my-low-level bash train.sh

# 可选的 5-keypoint locomotion policy。
PIPELINE=5kp LOW_WANDB_PROJECT=my-low-level bash train.sh

# 单 GPU：
CUDA_VISIBLE_DEVICES=0 NPROC=1 PIPELINE=stiff30 \
  LOW_WANDB_PROJECT=my-low-level bash train.sh
```

对应的 task 配置为：

```text
cfg/task/G1/G1_3kp_stiff.yaml
cfg/task/G1/G1_3kp_ee_net_pull_force_b.yaml
cfg/task/G1/G1_3kp_ee_xyz_range_100_600.yaml
cfg/task/G1/G1_5kp.yaml
```

训练完成后，脚本会打印高层训练所需的 run path：

```text
LOW_RUN_PATH=<entity>/<project>/compliance_3kp_stiff_finetune_limmt_full_force30_<timestamp>
```

若从本地 checkpoint 开始训练，可将 `checkpoint_path=/path/to/checkpoint.pt` 传给 `scripts/train.py`。

