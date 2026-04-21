#!/bin/bash

# ======== 1. 配置欲评估的模型列表 ========
MODELS=(
    "/root/autodl-fs/model/math_step_60_grpo_penalized"
    # "/root/autodl-fs/model/another_model_path"
)

# ======== 2. 配置欲评估的数据集列表 (格式: 数据集路径:Split) ========
DATASETS=(
    "HuggingFaceH4/MATH-500:test"
    "math-ai/aime24:test"
    "AI-MO/aimo-validation-amc:train"
    "math-ai/aime25:test"
)

# ======== 3. 评估参数设置 ========
NUM_GPUS=2              # 使用显卡数量
NUM_GENERATION=16       # 每个问题的生成数量 (Pass@16)
TEMPERATURE=0.6         # 采样温度
BATCH_SIZE=400          # 批处理大小 (若 OOM 请调小)
MAX_TOKENS=2048         # 单次生成最大长度
BASE_OUTPUT_DIR="./results"
mkdir -p "$BASE_OUTPUT_DIR"

# ======== 4. 执行循环评估 ========
for MODEL_PATH in "${MODELS[@]}"; do
    # 检查模型路径是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model path $MODEL_PATH not found, skipping..."
        continue
    fi

    MODEL_NAME=$(basename "$MODEL_PATH")
    MODEL_SUMMARY="${BASE_OUTPUT_DIR}/${MODEL_NAME}/summary.txt"
    mkdir -p "${BASE_OUTPUT_DIR}/${MODEL_NAME}"
    
    echo "===== Evaluation Summary for $MODEL_NAME =====" > "$MODEL_SUMMARY"
    date >> "$MODEL_SUMMARY"
    echo "==============================================" >> "$MODEL_SUMMARY"
    
    for ENTRY in "${DATASETS[@]}"; do
        # 拆分数据集路径和 Split
        IFS=':' read -r DATASET DS_SPLIT <<< "$ENTRY"

        # 清理数据集名称中的斜杠，用于创建目录
        SAFE_DS_NAME=$(echo "$DATASET" | sed 's/\//_/g')
        CURRENT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}/${SAFE_DS_NAME}_${DS_SPLIT}"
        mkdir -p "$CURRENT_OUTPUT_DIR"

        echo "--------------------------------------------------------"
        echo "Processing: Model=[$MODEL_NAME] | Dataset=[$DATASET] | Split=[$DS_SPLIT]"
        echo "Log saved to: $CURRENT_OUTPUT_DIR"
        echo "--------------------------------------------------------"

        # 1. 运行 eval.py 生成预测结果并评分
        python eval.py \
          --model_name="$MODEL_PATH" \
          --datasets="$DATASET" \
          --split="$DS_SPLIT" \
          --output_dir="$CURRENT_OUTPUT_DIR" \
          --max_tokens=$MAX_TOKENS \
          --num_gpus=$NUM_GPUS \
          --temperature=$TEMPERATURE \
          --num_generation=$NUM_GENERATION

        # 2. 自动计算指标
        GENERATED_FILE=$(ls -t "$CURRENT_OUTPUT_DIR"/*.jsonl | head -n 1)
        
        if [ -f "$GENERATED_FILE" ]; then
            echo "[Metrics Result] $MODEL_NAME | $DATASET ($DS_SPLIT):"
            python calculate_metrics.py \
                --file_path "$GENERATED_FILE" \
                --n_samples $NUM_GENERATION \
                --summary_file "$MODEL_SUMMARY" \
                --model_name "$MODEL_NAME" \
                --dataset_name "$DATASET"
        else
            echo "Error: No result file found for $MODEL_NAME on $DATASET"
        fi
        echo "Done!"
        echo ""
    done
done

echo "All evaluations completed! Check summary.txt in each model's directory for results."