#!/bin/bash

OUT_FILE="summary.json"
TMP_FILE="summary_tmp.json"

echo "[" > "$TMP_FILE"

SCENARIOS=(
    unitree_g1_pack_camera
    unitree_z1_dual_arm_cleanup_pencils
    unitree_z1_dual_arm_stackbox
    unitree_z1_dual_arm_stackbox_v2
    unitree_z1_stackbox
)

CASES=(case1)

FIRST=1
for scenario in "${SCENARIOS[@]}"; do
    for case in "${CASES[@]}"; do
        RESULT_FILE="${scenario}/${case}/result.json"
        if [ -f "$RESULT_FILE" ]; then
            if [ $FIRST -eq 1 ]; then
                jq 'del(.time_before_optim)' "$RESULT_FILE" >> "$TMP_FILE"
                FIRST=0
            else
                echo "," >> "$TMP_FILE"
                jq 'del(.time_before_optim)' "$RESULT_FILE" >> "$TMP_FILE"
            fi
        else
            echo "Warning: $RESULT_FILE not found, skipped."
        fi
    done
done

echo "]" >> "$TMP_FILE"
mv "$TMP_FILE" "$OUT_FILE"
echo "Summary saved to $OUT_FILE"