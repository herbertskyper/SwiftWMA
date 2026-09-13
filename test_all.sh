#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
SCENARIOS=(
    unitree_g1_pack_camera
    unitree_z1_dual_arm_cleanup_pencils
    unitree_z1_dual_arm_stackbox
    unitree_z1_dual_arm_stackbox_v2
    unitree_z1_stackbox
)

CASES=(case1)

for scenario in "${SCENARIOS[@]}"; do
    for case in "${CASES[@]}"; do
        echo "=============================="
        echo "Testing $scenario/$case"
        ./test.sh "$scenario" "$case"
        echo "------------------------------"
    done
done