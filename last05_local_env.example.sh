#!/usr/bin/env bash
# Copy this file to last05_local_env.sh and edit it for each machine.
# The main train/eval scripts auto-source last05_local_env.sh from either:
#   1) $LAST05_LOCAL_ENV
#   2) the parent last05_beta directory
#   3) this repository root

_LAST05_LOCAL_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAST05_BETA_ROOT="${LAST05_BETA_ROOT:-$(cd "${_LAST05_LOCAL_ENV_DIR}/.." && pwd)}"

export DATABASE_ROOT="${DATABASE_ROOT:-/data/database}"
export EXPERIMENTS_LIBERO_ROOT="${EXPERIMENTS_LIBERO_ROOT:-/data/experiments}"
export EXPERIMENTS_RLBENCH_ROOT="${EXPERIMENTS_RLBENCH_ROOT:-/data/experiments_rlbench}"

export REQUIRES_ROOT="${REQUIRES_ROOT:-/data/requires}"
export LIFT3D_ROOT="${LIFT3D_ROOT:-${REQUIRES_ROOT}/LIFT3D}"
export COSMOS_ROOT="${COSMOS_ROOT:-/data/experiments}"
export LIBERO_ROOT="${LIBERO_ROOT:-/data/LIBERO}"
export PYREP_PYTHON_PATH="${PYREP_PYTHON_PATH:-/data/python_pkgs}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/data/CoppeliaSim}"
export PORTABLE_XVFB_ROOT="${PORTABLE_XVFB_ROOT:-/data/train/bash/portable_xvfb}"

export CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-last05_qwen3vl}"
export CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}}"
