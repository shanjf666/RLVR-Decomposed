#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# ======== 1. 配置欲评估的模型列表 ========
MODELS=(
    # "/root/autodl-fs/model/math_step_45_update_both"
    "/root/autodl-tmp/data/models/modelscope_cache/models/lijia321/test-prompt-amc-step30"
    "/root/autodl-tmp/data/models/modelscope_cache/models/shanjf/dapo_final_step30"
    "/root/autodl-tmp/data/models/modelscope_cache/models/lijia321/test-math-candiate-normed_step60"
)

# ======== 2. 配置欲评估的数据集列表 (格式: 数据集路径:Split) ========
DATASETS=(
    "HuggingFaceH4/MATH-500:test"
    "math-ai/aime24:test"
    "AI-MO/aimo-validation-amc:train"
    "math-ai/aime25:test"
)

# ======== 3. 评估参数设置 ========
NUM_GPUS=4
NUM_GENERATION=32
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS=8192
BASE_OUTPUT_DIR="./results"

mkdir -p "$BASE_OUTPUT_DIR"

# ======== 4. 执行循环评估 ========
for MODEL_PATH in "${MODELS[@]}"; do
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model path $MODEL_PATH not found, skipping..."
        continue
    fi

    MODEL_NAME=$(basename "$MODEL_PATH")
    MODEL_SUMMARY="${BASE_OUTPUT_DIR}/${MODEL_NAME}/summary.txt"
    MODEL_SUMMARY_JSON="${BASE_OUTPUT_DIR}/${MODEL_NAME}/summary.json"
    mkdir -p "${BASE_OUTPUT_DIR}/${MODEL_NAME}"
    
    if [ ! -f "$MODEL_SUMMARY" ]; then
        echo "===== Evaluation Summary for $MODEL_NAME =====" > "$MODEL_SUMMARY"
        date >> "$MODEL_SUMMARY"
        echo "==============================================" >> "$MODEL_SUMMARY"
    else
        echo "" >> "$MODEL_SUMMARY"
        echo "--- Evaluation Resumed on $(date) ---" >> "$MODEL_SUMMARY"
    fi
    
    for ENTRY in "${DATASETS[@]}"; do
        # 拆分数据集路径和 Split
        DATASET=$(echo $ENTRY | cut -d':' -f1)
        DS_SPLIT=$(echo $ENTRY | cut -d':' -f2)

        SAFE_DS_NAME=$(echo "$DATASET" | sed 's/\//_/g')
        CURRENT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}/${SAFE_DS_NAME}_${DS_SPLIT}"
        mkdir -p "$CURRENT_OUTPUT_DIR"

        echo "--------------------------------------------------------"
        echo "Processing: Model=[$MODEL_NAME] | Dataset=[$DATASET] | Split=[$DS_SPLIT]"
        echo "--------------------------------------------------------"

        # 1. 运行数据并行推理
        PIDS=()
        for (( i=0; i<$NUM_GPUS; i++ )); do
            CUDA_VISIBLE_DEVICES=$i python eval.py \
              --model_name="$MODEL_PATH" \
              --datasets="$DATASET" \
              --split="$DS_SPLIT" \
              --output_dir="$CURRENT_OUTPUT_DIR" \
              --max_tokens=$MAX_TOKENS \
              --num_gpus=1 \
              --temperature=$TEMPERATURE \
              --top_p=$TOP_P \
              --top_k=$TOP_K \
              --num_generation=$NUM_GENERATION \
              --num_shards=$NUM_GPUS \
              --shard_id=$i &
            PIDS+=($!)
        done

        for pid in "${PIDS[@]}"; do
            wait $pid
        done

        # 合并结果
        GENERATED_FILE="$CURRENT_OUTPUT_DIR/merged_results.jsonl"
        cat "$CURRENT_OUTPUT_DIR"/*-shard_*.jsonl > "$GENERATED_FILE" 2>/dev/null
        rm -f "$CURRENT_OUTPUT_DIR"/*-shard_*.jsonl

        # 2. 计算指标
        if [ -s "$GENERATED_FILE" ]; then
            python calculate_metrics.py \
                --file_path "$GENERATED_FILE" \
                --n_samples $NUM_GENERATION \
                --summary_file "$MODEL_SUMMARY" \
                --summary_json "$MODEL_SUMMARY_JSON" \
                --model_name "$MODEL_NAME" \
                --dataset_name "$DATASET"
        else
            echo "Error: No results found for $DATASET"
        fi
    done
done

echo "All evaluations completed!"