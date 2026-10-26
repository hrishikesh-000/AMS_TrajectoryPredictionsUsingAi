#!/bin/bash
# =============================================================================
# run_pipeline.slurm
#
# End-to-end pipeline: train GNS from scratch on FINAL_OUTPUT/dataset ->
# sweep saved checkpoints against test.npz to pick the best one by rollout
# RMSE (no valid.npz is used, by request; this sweep is the substitute for
# early stopping) -> run a plain GNS rollout AND an EnKF-corrected rollout
# with the best checkpoint -> compute RMSE/MSE/R2 before vs after EnKF ->
# render the 3-panel comparison gif -> write final_results.json/.txt.
#
# Submit with:  sbatch run_pipeline.slurm
# It is requeue-safe: if training is interrupted/times out, resubmitting
# the SAME job resumes from the latest saved checkpoint instead of
# restarting from step 0 (gns.train's own --model_file=latest mechanism).
#
# ANTS cluster (ants.cs.ovgu.de): matches the working interactive command
# `srun -p gpu-stud --gpus=h100:1 --pty bash` -- gpu-stud partition, H100 via
# --gpus (not the generic --gres=gpu:1), no --account needed.
# =============================================================================
#SBATCH --job-name=gns_final_pipeline
#SBATCH --output=/home/iqaw35og/GNS_Project/FINAL_OUTPUT/logs/%x_%j.out
#SBATCH --error=/home/iqaw35og/GNS_Project/FINAL_OUTPUT/logs/%x_%j.err
#SBATCH --partition=gpu-stud
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:00
# If your cluster supports it, uncomment to auto-requeue on timeout:
# #SBATCH --requeue
# #SBATCH --signal=B:USR1@120

set -euo pipefail

# --------------------------------------------------------------------------
# Paths -- adjust ROOT if FINAL_OUTPUT isn't at this location on the cluster.
# GNS_REPO_ROOT must point at your geoelements/gns checkout (the one with
# gns/train.py in it).
# --------------------------------------------------------------------------
ROOT="${ROOT:-$HOME/GNS_Project/FINAL_OUTPUT}"
export GNS_REPO_ROOT="${GNS_REPO_ROOT:-$HOME/gns}"

DATA_PATH="$ROOT/dataset/"
MODEL_PATH="$ROOT/models/"
TRAIN_ROLLOUT_OUT="$ROOT/train_mode_rollout_scratch/"   # gns.train --mode=train needs an output_path even though it's unused for training itself
CKPT_EVAL_DIR="$ROOT/checkpoint_eval/"
ENKF_RESULTS_DIR="$ROOT/results/enkf_rollout/"
GIF_DIR="$ROOT/results/gifs/"
LOG_DIR="$ROOT/logs/"
RESULTS_DIR="$ROOT/results/"
SCRIPTS_DIR="$ROOT/scripts/"

mkdir -p "$MODEL_PATH" "$TRAIN_ROLLOUT_OUT" "$CKPT_EVAL_DIR" "$ENKF_RESULTS_DIR" "$GIF_DIR" "$LOG_DIR" "$RESULTS_DIR"

# --------------------------------------------------------------------------
# Hyperparameters. User explicitly OK'd increasing both epochs and training
# set vs. the previous run (which was 200,000 steps over a 120-frame / 114
# unique-window train.npz -- ~3,500 epochs of repetition on one trajectory).
# New train.npz has 260 frames / 254 unique windows (dataset now also spans
# both the active-flow AND settled regimes, see build_final_dataset.py).
# 1,000,000 steps here is a deliberate, larger-but-not-reckless step up:
# checkpointed every 25,000 steps so eval_checkpoints.py can pick the point
# of best ROLLOUT quality rather than just taking the final, most-overfit
# step.
# --------------------------------------------------------------------------
NTRAIN_STEPS="${NTRAIN_STEPS:-1000000}"
NSAVE_STEPS="${NSAVE_STEPS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
# Full LR decay cycle (down to 10% of lr_init) across the whole run, instead
# of the library default (5,000,000) which barely decays at all over a run
# this length.
LR_DECAY_STEPS="$NTRAIN_STEPS"

# EnKF settings for the FINAL before/after comparison. obs_fraction=0.1
# (10% of particles observed) rather than the script's own default of 1.0:
# observing 100% of particles every few steps means periodically handing
# the filter the whole true state, which flatters "after" numbers without
# proving the correction generalizes. 10% is a more honest sparse-sensor
# test. Change these via env vars if you want a different comparison.
N_ENSEMBLE="${N_ENSEMBLE:-30}"
ASSIM_EVERY="${ASSIM_EVERY:-10}"
OBS_FRACTION="${OBS_FRACTION:-0.1}"
OBS_NOISE_STD="${OBS_NOISE_STD:-1e-4}"
INIT_PERTURB_STD="${INIT_PERTURB_STD:-1e-4}"

echo "=== Environment ==="
echo "ROOT=$ROOT"
echo "GNS_REPO_ROOT=$GNS_REPO_ROOT"
echo "NTRAIN_STEPS=$NTRAIN_STEPS NSAVE_STEPS=$NSAVE_STEPS BATCH_SIZE=$BATCH_SIZE"

# --------------------------------------------------------------------------
# ANTS-specific: Spack provides the toolchain (CUDA etc.) the venv's
# GPU-enabled torch build needs -- must be sourced before the venv activates.
# Then activate the venv (not conda -- ANTS uses a plain Python venv).
# Adjust GNS_VENV_PATH if yours lives somewhere other than ~/gns_env.
# --------------------------------------------------------------------------
source /opt/spack/main/env.sh
spack load /wun5tnb
GNS_VENV_PATH="${GNS_VENV_PATH:-$HOME/gns_env}"
source "$GNS_VENV_PATH/bin/activate"
PYTHON_EXE="$(which python)"

# --------------------------------------------------------------------------
# gns/train.py calls torch.distributed.init_process_group(backend="nccl", ...)
# even for a single-GPU run (world_size=1) -- NCCL's automatic network-
# interface discovery is known to hang on multi-NIC HPC nodes even when
# there's only one process with nothing to actually communicate with.
# Force it to use loopback and skip IB/P2P negotiation entirely.
# Also disable Python's stdout buffering so log output appears in real time
# instead of sitting in a block buffer once redirected to a file.
# --------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
echo "python: $PYTHON_EXE"

# ==========================================================================
# STAGE 1: TRAIN (resumes automatically if checkpoints already exist)
# ==========================================================================
TRAIN_TIME_FILE="$RESULTS_DIR/training_time_seconds.txt"
LATEST_CKPT=$(ls "$MODEL_PATH"/model-*.pt 2>/dev/null | sort -V | tail -n1 || true)

RESUME_ARGS=()
if [ -n "$LATEST_CKPT" ]; then
    echo "Found existing checkpoint(s) in $MODEL_PATH -- resuming training."
    RESUME_ARGS=(--model_file=latest --train_state_file=latest)
else
    echo "No existing checkpoints -- training from scratch."
fi

TRAIN_START=$(date +%s)
cd "$GNS_REPO_ROOT"
python -m gns.train \
    --mode=train \
    --data_path="$DATA_PATH" \
    --model_path="$MODEL_PATH" \
    --output_path="$TRAIN_ROLLOUT_OUT" \
    --ntraining_steps="$NTRAIN_STEPS" \
    --nsave_steps="$NSAVE_STEPS" \
    --batch_size="$BATCH_SIZE" \
    --lr_decay_steps="$LR_DECAY_STEPS" \
    "${RESUME_ARGS[@]}"
TRAIN_END=$(date +%s)
THIS_RUN_SECONDS=$((TRAIN_END - TRAIN_START))

PREV_SECONDS=0
if [ -f "$TRAIN_TIME_FILE" ]; then
    PREV_SECONDS=$(cat "$TRAIN_TIME_FILE")
fi
TOTAL_SECONDS=$(python -c "print(float('$PREV_SECONDS') + float('$THIS_RUN_SECONDS'))")
echo "$TOTAL_SECONDS" > "$TRAIN_TIME_FILE"
echo "Training stage took ${THIS_RUN_SECONDS}s this run, ${TOTAL_SECONDS}s total (accumulated across resumes) -> $TRAIN_TIME_FILE"

# ==========================================================================
# STAGE 2: CHECKPOINT SWEEP -- pick the best checkpoint by test-rollout RMSE
# ==========================================================================
python "$SCRIPTS_DIR/eval_checkpoints.py" \
    --gns_repo_root="$GNS_REPO_ROOT" \
    --data_path="$DATA_PATH" \
    --model_path="$MODEL_PATH" \
    --output_dir="$CKPT_EVAL_DIR" \
    --python_exe="$PYTHON_EXE"

BEST_MODEL_FILE=$(python -c "import json; print(json.load(open('$CKPT_EVAL_DIR/best_checkpoint.json'))['model_file'])")
echo "Selected checkpoint: $BEST_MODEL_FILE"

# ==========================================================================
# STAGE 3: PLAIN GNS ROLLOUT (the "before") using the selected checkpoint
# ==========================================================================
cd "$SCRIPTS_DIR"
python gns_enkf_rollout.py \
    --data_path="$DATA_PATH" \
    --model_path="$MODEL_PATH" \
    --model_file="$BEST_MODEL_FILE" \
    --output_path="$ENKF_RESULTS_DIR" \
    --mode=plain

# ==========================================================================
# STAGE 4: EnKF-CORRECTED ROLLOUT (the "after") using the same checkpoint
# ==========================================================================
python gns_enkf_rollout.py \
    --data_path="$DATA_PATH" \
    --model_path="$MODEL_PATH" \
    --model_file="$BEST_MODEL_FILE" \
    --output_path="$ENKF_RESULTS_DIR" \
    --mode=enkf \
    --n_ensemble="$N_ENSEMBLE" \
    --assim_every="$ASSIM_EVERY" \
    --obs_fraction="$OBS_FRACTION" \
    --obs_noise_std="$OBS_NOISE_STD" \
    --init_perturb_std="$INIT_PERTURB_STD"

# ==========================================================================
# STAGE 5: FINAL METRICS REPORT (RMSE/MSE/R2 before vs after, training time)
# ==========================================================================
python final_report.py \
    --enkf_results_dir="$ENKF_RESULTS_DIR" \
    --best_checkpoint_json="$CKPT_EVAL_DIR/best_checkpoint.json" \
    --training_time_file="$TRAIN_TIME_FILE" \
    --output_dir="$RESULTS_DIR"

# ==========================================================================
# STAGE 6: COMPARISON GIF (Reality | Before/plain GNS | After/EnKF)
# ==========================================================================
python make_comparison_gif.py \
    --results_dir="$ENKF_RESULTS_DIR" \
    --out_dir="$GIF_DIR"

echo "=== DONE ==="
echo "Final results: $RESULTS_DIR/final_results.txt"
echo "Comparison gif: $GIF_DIR/enkf_comparison.gif"
