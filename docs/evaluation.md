[← README](../README.md)

## Baselines and evaluation

Evaluate a W&B policy:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --run_path <entity>/<project>/<run_id> \
  --ee-compliance-stiffness 200 200 200
```

Evaluate a local checkpoint with its training config (`cfg.yaml`, saved next to
the checkpoint):

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --checkpoint /path/to/checkpoint.pt \
  --config-file /path/to/cfg.yaml \
  --ee-compliance-stiffness 200 200 200
```

The principal baseline modes are:

| Baseline | Training or evaluation entry point |
|---|---|
| Analytical MoE | `--moe-experts-config cfg/moe/...yaml --low-run-path ...` |
| Stiff.-conditioned high-level | `RUN_ONLY=baselines bash train_hl_teacher_student.sh`, ranged HL task |
| Stiff.-conditioned end-to-end | `PIPELINE=range_100_600 bash train.sh` |
| Fixed end-to-end | `PIPELINE=fixed_ee bash train.sh` |
| Raw stiff low-level | `PIPELINE=stiff30 bash train.sh` |
| Estimated-force analytical compliance | Add `--ee-compliance-force-estimator-ablation` when evaluating a **high-level** EE policy |
| Oracle-force analytical compliance | Add `--ee-compliance-oracle-force` when evaluating the **stiff low-level** checkpoint |

Both force modes set the EE target to force / commanded stiffness; the oracle
mode uses the applied force, the estimator mode uses the policy's force
estimate.

Example oracle-force evaluation:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-compliance-eval \
  --ee-compliance-num-envs 8 \
  --run_path <entity>/<low-level-project>/<stiff_low_level_finetune_run> \
  --ee-compliance-oracle-force \
  --ee-compliance-stiffness 300 450 600
```

### Bimanual evaluation

Use `--ee-bimanual-compliance-eval` instead of `--ee-compliance-eval`, with six
stiffness values (`left_xyz right_xyz`) and a per-EE task:

```bash
python scripts/eval_manipulation.py \
  --full_collision \
  --ee-bimanual-compliance-eval \
  --ee-compliance-num-envs 8 \
  --moe-experts-config \
    cfg/moe/G1_ee_analytical_100_200_400_600_xyz600_direct_posscale035.yaml \
  --task G1/hl/ee/G1_hl_ee_bimanual_analytical_moe_200_600_force_b_student \
  --low-run-path <entity>/<low-level-project>/<low_level_finetune_run> \
  --ee-compliance-stiffness 200 400 600 600 400 200
```

## Evaluation protocol and outputs

The EE compliance evaluator applies forces of 5, 10, 15, 20, and 30 N along
`+/-x`, `+/-y`, `+/-z` and computes the apparent stiffness from the steady-state
displacement.

Other evaluation modes:

```bash
# EE tracking with scripted target poses.
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --ee-tracking-eval \
  --num_envs 8

# Root compliance evaluation.
python scripts/eval_manipulation.py \
  --run_path <entity>/<project>/<run_id> \
  --root-compliance-eval \
  --root-compliance-num-envs 8
```

Useful options:

| Option | Meaning |
|---|---|
| `--seed N` | Override the evaluation seed |
| `-i N` | Evaluate `checkpoint_N.pt` of a W&B run instead of the latest |
| `--ee-output PATH` / `--root-output PATH` | Explicit report path |

Reports are written as JSON under `outputs/`. If Isaac Sim hangs on exit after
printing the results, press Ctrl+C; the report is already saved.

## Reproducibility notes

- Compare policies with the same task config, low-level checkpoint, number of
  environments, and stiffness command.
- Set the evaluation seed with `--seed` and the training seed with `seed=N`.
- Keep each high-level checkpoint paired with the low-level checkpoint it was
  trained on.

