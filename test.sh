#!/bin/bash
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
# 检查参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <scenario_name> <case_id>"
    echo "Example: $0 unitree_g1_pack_camera case1"
    exit 1
fi

SCENARIO_NAME="$1"
CASE_ID="$2"

# 构造路径变量
BASE_DIR="${SCENARIO_NAME}/${CASE_ID}"
RUN_SCRIPT="${BASE_DIR}/run_world_model_interaction.sh"
LOG_FILE="${BASE_DIR}/output.log"
GT_VIDEO="${BASE_DIR}/${SCENARIO_NAME}_${CASE_ID}.mp4"
INFER_DIR="${BASE_DIR}/output/inference"
PSNR_JSON="${BASE_DIR}/psnr_result.json"
FINAL_RESULT_JSON="${BASE_DIR}/result.json"
BASELINE_FILE="./baseline.txt"

# 1. 执行 run_world_model_interaction.sh
echo ">>> Running interaction script: ${RUN_SCRIPT}"
if [ -f "$RUN_SCRIPT" ]; then
    bash "$RUN_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "Error: Execution of $RUN_SCRIPT failed."
        exit 1
    fi
else
    echo "Error: Script $RUN_SCRIPT not found."
    exit 1
fi

PRED_VIDEO=$(find "$INFER_DIR" -maxdepth 1 -type f -name "*.mp4" | head -n 1)
# 2. 运行 psnr_score_for_challenge.py
echo ">>> Calculating PSNR..."
python3 psnr_score_for_challenge.py \
    --gt_video "$GT_VIDEO" \
    --pred_video "$PRED_VIDEO" \
    --output_file "$PSNR_JSON"

if [ $? -ne 0 ]; then
    echo "Error: PSNR calculation failed."
    exit 1
fi

# 3. 提取运行时间 (Inference for sample ... completed in ... seconds)
echo ">>> Extracting execution time from log..."
# 从 output.log 中提取 'Inference for sample ... completed in ... seconds.' 这句的时间
INFER_TIME_LINE=$(grep "Inference for sample" "$LOG_FILE" | grep "completed in" | tail -n 1)
if [ -z "$INFER_TIME_LINE" ]; then
    echo "Error: Could not find inference time line in $LOG_FILE"
    exit 1
fi
# 提取秒数（假设格式为 ... completed in XX.XX seconds.）
TIME_VAL_1=$(echo "$INFER_TIME_LINE" | sed -E 's/.*completed in ([0-9.]+) seconds.*/\1/')
echo "Extracted Time: $TIME_VAL_1 s"

# 3.1 提取 MACs (如果日志中有输出)
MACS_LINE=$(grep "MACs for sample" "$LOG_FILE" | tail -n 1)
MACS_VAL=""
if [ -n "$MACS_LINE" ]; then
    MACS_VAL=$(echo "$MACS_LINE" | sed -E 's/.*: ([0-9.]+) T.*/\1/')
    echo "Extracted MACs: $MACS_VAL T"
fi

# 4. 从 psnr_result.json 提取 PSNR (值2)
echo ">>> Extracting PSNR value..."
if [ -f "$PSNR_JSON" ]; then
    # 使用 python 读取 json 并保留 4 位小数
    PSNR_VAL_2=$(python3 -c "import json; print(f'{json.load(open(\"$PSNR_JSON\"))[\"psnr\"]:.4f}')")
    echo "Extracted PSNR: $PSNR_VAL_2"
else
    echo "Error: $PSNR_JSON not found."
    exit 1
fi

# 5. 处理 baseline.txt (值3)
# 格式为: scenario_name/case_id:value
KEY="${SCENARIO_NAME}/${CASE_ID}"
TIME_VAL_3=""

if [ ! -f "$BASELINE_FILE" ]; then
    touch "$BASELINE_FILE"
fi

# 尝试在 baseline.txt 中查找
EXISTING_LINE=$(grep "^${KEY}:" "$BASELINE_FILE")

if [ -n "$EXISTING_LINE" ]; then
    # 提取冒号后的值
    TIME_VAL_3=$(echo "$EXISTING_LINE" | cut -d':' -f2 | xargs) # xargs 去除空格
    echo "Found baseline time: $TIME_VAL_3"
else
    # 如果不存在，使用当前运行时间作为 baseline，并追加到文件
    TIME_VAL_3="$TIME_VAL_1"
    echo "${KEY}:${TIME_VAL_3}" >> "$BASELINE_FILE"
    echo "Baseline not found. Added current time ($TIME_VAL_3) to $BASELINE_FILE"
fi

# 6. 生成最终结果 JSON
# 提取 case_id 的最后一位数字 (例如 case1 -> 1)
CASE_NUM=$(echo "$CASE_ID" | grep -o '[0-9]\+$' | tail -c 2) 
# 注意：如果 case_id 只有一位数，grep -o '[0-9]\+$' 就能拿到。如果是 'case1'，拿到的就是 '1'。
# 这里简单处理假设 case_id 结尾是数字
CASE_NUM=$(echo "$CASE_ID" | sed 's/[^0-9]*//g' | awk '{print substr($0, length($0), 1)}')

echo ">>> Generating result.json..."
if [ -n "$MACS_VAL" ]; then
python3 -c "
import json, os
path = \"$FINAL_RESULT_JSON\"
if os.path.exists(path):
    with open(path, 'r') as f:
        output = json.load(f)
else:
    output = {
        \"scenario\": \"$SCENARIO_NAME\",
        \"case_id\": $CASE_NUM,
        \"time_before_optim\": $TIME_VAL_3,
        \"time_after_optim\": $TIME_VAL_1,
        \"psnr\": $PSNR_VAL_2
    }
output[\"macs\"] = float(\"$MACS_VAL\")
with open(path, 'w') as f:
    json.dump(output, f, indent=4)
"
else
python3 -c "
import json
output = {
    \"scenario\": \"$SCENARIO_NAME\",
    \"case_id\": $CASE_NUM,
    \"time_before_optim\": $TIME_VAL_3,
    \"time_after_optim\": $TIME_VAL_1,
    \"psnr\": $PSNR_VAL_2
}
with open(\"$FINAL_RESULT_JSON\", 'w') as f:
    json.dump(output, f, indent=4)
"
fi

echo ">>> Done. Result saved to $FINAL_RESULT_JSON"
cat "$FINAL_RESULT_JSON"