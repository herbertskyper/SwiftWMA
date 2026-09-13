#!/bin/bash
set -euo pipefail

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# 遍历所有 <scenario>/<case*> 目录
for BASE_DIR in */case*; do
    [ -d "$BASE_DIR" ] || continue
    # BASE_DIR 形如 "scenario/case1"
    SCENARIO="${BASE_DIR%%/*}"
    CASE="${BASE_DIR#*/}"
    DEST_DIR="${RESULTS_DIR}/${SCENARIO}/${CASE}"
    mkdir -p "$DEST_DIR"

    LOG_FILE="${BASE_DIR}/output.log"
    INFER_DIR="${BASE_DIR}/output/inference"

    if [ -f "$LOG_FILE" ]; then
        cp -v "$LOG_FILE" "$DEST_DIR/"
    else
        echo "Warning: ${LOG_FILE} not found."
    fi

    if [ -d "$INFER_DIR" ]; then
        mapfile -t MP4_FILES < <(find "$INFER_DIR" -maxdepth 1 -type f -name "*.mp4" | sort)
        if [ "${#MP4_FILES[@]}" -eq 1 ]; then
            cp -v "${MP4_FILES[0]}" "$DEST_DIR/"
        elif [ "${#MP4_FILES[@]}" -eq 0 ]; then
            echo "Warning: no top-level mp4 found under ${INFER_DIR}."
        else
            echo "Warning: expected unique top-level mp4 under ${INFER_DIR}, found ${#MP4_FILES[@]}; skip copy to avoid wrong file."
        fi
    else
        echo "Warning: ${INFER_DIR} not found."
    fi
done

echo "Gather complete. Results are under: ${RESULTS_DIR}/"