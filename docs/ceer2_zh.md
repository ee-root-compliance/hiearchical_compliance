[← 中文首页](../README_zh.md)

## 高层 teacher-student 训练

`train_hl_teacher_student.sh` 在冻结的低层 checkpoint 之上训练高层 EE policy，包含两个阶段：

```text
teacher     root_student_force_ppo
adapt       root_student_force_ppo_adapt
```

训练 100--600 N/m 解析 MoE 使用的十个高层 experts：

```bash
export LOW_RUN_PATH=<entity>/<project>/compliance_3kp_stiff_finetune_limmt_full_force30_<timestamp>
export HL_WANDB_PROJECT=my-high-level
RUN_ONLY=experts bash train_hl_teacher_student.sh
```

十个 expert 的目标为：

```text
100x, 100y, 100z,
200x, 200y, 200z,
400x, 400y, 400z,
600xyz。
```

训练匹配的高层 baseline，包括固定 200xyz policy 和 100--600 ranged high-level policy：

```bash
RUN_ONLY=baselines bash train_hl_teacher_student.sh
```

对应 task 配置位于 `cfg/task/G1/hl/ee/`。Root compliance task 位于 `cfg/task/G1/hl/root/`；如需训练，请取消 `train_hl_teacher_student.sh` 中对应行的注释。

## 解析 MoE

解析 MoE 不需要训练。它冻结十个 expert checkpoint，在物理位移空间中插值，并独立组合 x、y、z 三个方向的 residual action。配置文件为：

```text
cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml
```

启动前需要填写 `experts.<name>` 中的十个 checkpoint：

```yaml
checkpoint_path: /path/to/checkpoint.pt
checkpoint_path: run:entity/project/run_id
checkpoint_path: run:entity/project/run_id:2000
run_path: entity/project/run_id
```

然后传入需要评测的 stiffness：

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --moe-experts-config \
    cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml \
  --low-run-path <entity>/<low-level-project>/<low_level_finetune_run> \
  --ee-compliance-stiffness 300 450 600
```

本地低层 checkpoint 使用 `--low-checkpoint /path/to/low_level.pt`。

