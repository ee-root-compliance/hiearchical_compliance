[← 中文首页](../README_zh.md)

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
| `PIPELINE` | `stiff30` | `stiff30`、`fixed_ee`、`range_100_600` 或 `5kp` |
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

