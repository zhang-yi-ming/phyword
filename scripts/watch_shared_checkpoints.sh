#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <shared_run_dir> <backup_root_dir> [poll_seconds] [settle_seconds]"
    echo "Example:"
    echo "  $0 /media/zhangyiming/last05/exp_cosmos_vla_cot_pre_6_2_1/cosmos2B_janus1B_3expert_cot_spatial_bs8_lr1e-4 /media/zhangyiming/last05/checkpoint_backups 20 300"
    exit 1
fi

SHARED_RUN_DIR="$1"
BACKUP_ROOT_DIR="$2"
POLL_SECONDS="${3:-20}"
SETTLE_SECONDS="${4:-300}"
STATE_DIR="${BACKUP_ROOT_DIR}/.watch_state"
LOG_FILE="${BACKUP_ROOT_DIR}/watch.log"

mkdir -p "${BACKUP_ROOT_DIR}" "${STATE_DIR}"

copy_checkpoint() {
    local src_dir="$1"
    local ckpt_name
    local stamp
    local dest_dir

    ckpt_name="$(basename "${src_dir}")"
    stamp="$(date '+%Y%m%d_%H%M%S')"
    dest_dir="${BACKUP_ROOT_DIR}/${ckpt_name}__snapshot_${stamp}"

    echo "[$(date '+%F %T')] Snapshot ${src_dir} -> ${dest_dir}" | tee -a "${LOG_FILE}"

    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete "${src_dir}/" "${dest_dir}/"
    else
        mkdir -p "${dest_dir}"
        cp -a "${src_dir}/." "${dest_dir}/"
    fi
}

record_value() {
    local file_path="$1"
    local value="$2"
    printf '%s\n' "${value}" > "${file_path}"
}

while true; do
    shopt -s nullglob
    checkpoint_dirs=("${SHARED_RUN_DIR}"/checkpoint-*)
    shopt -u nullglob

    for ckpt_dir in "${checkpoint_dirs[@]}"; do
        [[ -d "${ckpt_dir}" ]] || continue

        ckpt_name="$(basename "${ckpt_dir}")"
        key_file="${STATE_DIR}/${ckpt_name}.state"
        pending_file="${STATE_DIR}/${ckpt_name}.pending"

        # Use directory mtime and model file size to detect new or overwritten checkpoints.
        model_file="${ckpt_dir}/cosmos_janus_mot.pt"
        dir_mtime="$(stat -c '%Y' "${ckpt_dir}" 2>/dev/null || echo 0)"
        model_size=0
        if [[ -f "${model_file}" ]]; then
            model_size="$(stat -c '%s' "${model_file}" 2>/dev/null || echo 0)"
        fi
        state_value="${dir_mtime}:${model_size}"

        previous_value=""
        if [[ -f "${key_file}" ]]; then
            previous_value="$(cat "${key_file}")"
        fi

        # Ignore empty/incomplete checkpoints until the model file exists.
        if [[ "${model_size}" == "0" ]]; then
            continue
        fi

        pending_since=""
        pending_value=""
        if [[ -f "${pending_file}" ]]; then
            IFS='|' read -r pending_since pending_value < "${pending_file}" || true
        fi

        if [[ "${state_value}" != "${previous_value}" ]]; then
            now_epoch="$(date +%s)"

            # Any newly observed change resets the settle timer.
            if [[ "${state_value}" != "${pending_value}" ]]; then
                record_value "${pending_file}" "${now_epoch}|${state_value}"
                echo "[$(date '+%F %T')] Change detected for ${ckpt_dir}, waiting ${SETTLE_SECONDS}s before snapshot." >> "${LOG_FILE}"
                continue
            fi

            if [[ -n "${pending_since}" ]] && (( now_epoch - pending_since >= SETTLE_SECONDS )); then
                copy_checkpoint "${ckpt_dir}"
                record_value "${key_file}" "${state_value}"
                rm -f "${pending_file}"
            fi
        else
            rm -f "${pending_file}"
        fi
    done

    sleep "${POLL_SECONDS}"
done
