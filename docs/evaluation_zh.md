[← 中文首页](../README_zh.md)

## Baseline 和评测

评测 W&B policy：

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --run_path <entity>/<project>/<run_id> \
  --ee-compliance-stiffness 200 200 200
```

主要 baseline：

| Baseline | 训练或评测入口 |
|---|---|
| Analytical MoE | `--moe-experts-config cfg/moe/...yaml --low-run-path ...` |
| Stiffness-conditioned high-level | `RUN_ONLY=baselines bash train_hl_teacher_student.sh`，使用 ranged HL task |
| Stiffness-conditioned end-to-end | `PIPELINE=range_100_600 bash train.sh` |
| Fixed end-to-end | `PIPELINE=fixed_ee bash train.sh` |
| Raw stiff low-level | `PIPELINE=stiff30 bash train.sh` |
| Estimated-force analytical compliance | 评测高层 EE policy 时添加 `--ee-compliance-force-estimator-ablation` |
| Oracle-force analytical compliance | 评测 stiff low-level checkpoint 时添加 `--ee-compliance-oracle-force` |

两个 force mode 都会根据 force / commanded stiffness 设置 EE target；oracle mode 使用施加的真实力，estimator mode 使用 policy 的力估计。

## 评测协议和输出

EE compliance evaluator 沿 `+/-x`、`+/-y`、`+/-z` 施加 5、10、15、20 和 30 N 的力，并根据稳态位移计算 apparent stiffness。

其他评测模式：

```bash
# 使用脚本目标 pose 评测 EE tracking。
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --ee-tracking-eval \
  --num_envs 8

# Root compliance 评测。
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --root-compliance-eval \
  --root-compliance-num-envs 8
```

常用选项：

| 选项 | 含义 |
|---|---|
| `--seed N` | 覆盖评测 seed |
| `-i N` | 使用 W&B run 的 `checkpoint_N.pt`，而不是最新 checkpoint |
| `--ee-output PATH` / `--root-output PATH` | 指定 JSON 报告路径 |

报告默认写入 `outputs/`。如果 Isaac Sim 打印结果后退出时卡住，可以按 Ctrl+C；报告已经保存。

## 可复现性说明

- 比较不同 policy 时，使用相同的 task config、低层 checkpoint、环境数量和 stiffness command。
- 使用 `--seed` 设置评测 seed，使用 `seed=N` 设置训练 seed。
- 确保每个高层 checkpoint 与训练它时使用的低层 checkpoint 匹配。

