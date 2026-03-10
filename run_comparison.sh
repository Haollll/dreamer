#!/usr/bin/env bash
# ============================================================
# Dreamer world-model comparison on Pendulum-v1
# Compatible with CPU / Colab.  Total wall time: ~1-3h on CPU.
#
# Usage:
#   bash run_comparison.sh
#
# Or run individual models:
#   bash run_comparison.sh rssm
#   bash run_comparison.sh linear
#   bash run_comparison.sh mlp
# ============================================================

set -e

TASK="gym_Pendulum-v1"
STEPS=50000
PREFILL=1000
TIME_LIMIT=200
ACTION_REPEAT=2
EVAL_EVERY=5000
SEED=42
PRECISION=32   # CPU-friendly; use 16 on GPU

RUN_RSSM=true
RUN_LINEAR=true
RUN_MLP=true

# Allow selective runs from CLI
if [ $# -gt 0 ]; then
  RUN_RSSM=false; RUN_LINEAR=false; RUN_MLP=false
  for arg in "$@"; do
    case $arg in
      rssm)   RUN_RSSM=true ;;
      linear) RUN_LINEAR=true ;;
      mlp)    RUN_MLP=true ;;
      *) echo "Unknown model: $arg (choose rssm|linear|mlp)"; exit 1 ;;
    esac
  done
fi

COMMON="--task $TASK \
        --steps $STEPS \
        --prefill $PREFILL \
        --time_limit $TIME_LIMIT \
        --action_repeat $ACTION_REPEAT \
        --eval_every $EVAL_EVERY \
        --seed $SEED \
        --precision $PRECISION \
        --log_images False"

# ---- RSSM (original Dreamer) ----
if $RUN_RSSM; then
  echo "=== Training RSSM ==="
  python dreamer.py $COMMON \
      --world_model rssm \
      --logdir logs/rssm
fi

# ---- LinearSSM ----
if $RUN_LINEAR; then
  echo "=== Training LinearSSM ==="
  python dreamer.py $COMMON \
      --world_model linear \
      --logdir logs/linear
fi

# ---- MLPDynamics ----
if $RUN_MLP; then
  echo "=== Training MLPDynamics ==="
  python dreamer.py $COMMON \
      --world_model mlp \
      --logdir logs/mlp
fi

# ---- Evaluate and plot ----
echo "=== Generating plots ==="
python evaluate.py \
    --logdirs  logs/rssm logs/linear logs/mlp \
    --labels   RSSM      LinearSSM   MLP \
    --world_models rssm  linear      mlp \
    --outdir   results \
    --horizon  5 \
    --mse_transitions 1000

echo "Done.  Plots saved to results/"
