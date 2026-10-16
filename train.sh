#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
# All values can be overridden from the environment, e.g.
#   WANDB_ENTITY=my-team PIPELINE=stiff30 ./train.sh
WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
LOW_WANDB_PROJECT="${LOW_WANDB_PROJECT:-your-low-level-project}"
PROJECT="${WANDB_ENTITY}/${LOW_WANDB_PROJECT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
# Differs from train_hl_teacher_student.sh's default so both can run at once.
MASTER_PORT="${MASTER_PORT:-29501}"
SCRIPT="scripts/train.py"

if [[ "$WANDB_ENTITY" == "your-wandb-entity" ]]; then
  echo "Error: set WANDB_ENTITY to your W&B user or team, e.g." >&2
  echo "  WANDB_ENTITY=my-team ./train.sh" >&2
  echo "Stages are chained through W&B run paths run:<entity>/<project>/<run_id>." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES

# Select a low-level pipeline explicitly when launching from tmux.  The public
# default is the 30 N non-compliant 3kp backbone used by the paper.
PIPELINE="${PIPELINE:-stiff30}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

run_pipeline() {
  local TASK="$1" TAG="$2" SUFFIX="$3"

  local ID_TRAIN="${TAG}_train_${SUFFIX}"
  local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
  local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

  # ---------- TRAIN ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=train
    wandb.entity="$WANDB_ENTITY" wandb.project="$LOW_WANDB_PROJECT"
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.entity="$WANDB_ENTITY" wandb.project="$LOW_WANDB_PROJECT"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.entity="$WANDB_ENTITY" wandb.project="$LOW_WANDB_PROJECT"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  echo
  echo "=== Low-level pipeline finished. Use this for train_hl_teacher_student.sh: ==="
  echo "LOW_RUN_PATH=${PROJECT}/${ID_FINETUNE}"
}

# The following low-level pipelines are the supported paper/public options:
#
#   PIPELINE=ceer1
#     Compliant 3-kp controller with legacy anchor-force/admittance targets.
#
#   PIPELINE=stiff30
#     Shared 3-kp stiff low-level policy with 30 N external-force probes.
#
#   PIPELINE=fixed_ee
#     End-to-end fixed 200xyz EE-compliance baseline.
#
#   PIPELINE=range_100_600
#     End-to-end independently ranged xyz compliance baseline.
#
#   PIPELINE=5kp
#     Optional 5-kp locomotion policy.
#
# Each option runs train -> adapt -> finetune.  Keep the alternatives here so
# the public repository has one low-level launcher without old one-off shell
# wrappers scattered across the root directory.
case "$PIPELINE" in
  ceer1)
    run_pipeline "G1/G1_3kp" "ceer1_3kp_compliant" \
      "${RUN_TIMESTAMP}"
    ;;
  stiff30)
    run_pipeline "G1/G1_3kp_stiff" "compliance_3kp_stiff" \
      "limmt_full_force30_${RUN_TIMESTAMP}"
    ;;
  fixed_ee)
    run_pipeline "G1/G1_3kp_ee_net_pull_force_b" "compliance_3kp_ee_fixed200" \
      "${RUN_TIMESTAMP}"
    ;;
  range_100_600)
    run_pipeline "G1/G1_3kp_ee_xyz_range_100_600" \
      "compliance_3kp_ee_xyz_range_100_600" "${RUN_TIMESTAMP}"
    ;;
  5kp)
    run_pipeline "G1/G1_5kp" "compliance_5kp" "${RUN_TIMESTAMP}"
    ;;
  *)
    echo "Unknown PIPELINE=$PIPELINE" >&2
    echo "Choose: ceer1, stiff30, fixed_ee, range_100_600, or 5kp" >&2
    exit 2
    ;;
esac
