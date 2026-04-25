#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# ======== 1. 配置欲评估的模型列表 ========
MODELS=(
    # "/root/autodl-fs/model/math_step_45_update_both"
    # "/root/autodl-tmp/data/models/modelscope_cache/models/lijia321/test-prompt-amc-step30"
    # "/root/autodl-tmp/data/models/modelscope_cache/models/shanjf/dapo_final_step30"
    # "/root/autodl-tmp/data/models/modelscope_cache/models/lijia321/test-math-candiate-normed_step60"
    # "/root/autodl-tmp/data/models/modelscope_cache/models/shanjf/Qwen3-4B-DIV-TTRL-then-pass1"
    # "/root/autodl-tmp/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base"
    "/root/autodl-tmp/data/models/modelscope_cache/models/shanjf/dapo_final_prompt_normalized_step75"
    # 你也可以直接写 Hugging Face 的 ID，例如：
    # "Qwen/Qwen2.5-Math-7B-Instruct"
)

# ======== 2. 配置欲评估的数据集列表 (格式: 数据集路径:Split) ========
DATASETS=(
    # "HuggingFaceH4/MATH-500:test"
    "math-ai/aime24:test"
    # "AI-MO/aimo-validation-amc:train"
    # "math-ai/aime25:test"
)

# ======== 3. 评估参数设置 ========
NUM_GPUS=4
TP_SIZE=2       # 每个模型实例使用的 GPU 数量 (Tensor Parallelism)。总副本数 = NUM_GPUS / TP_SIZE
NUM_GENERATION=16
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS=8192
BASE_OUTPUT_DIR="./results"

mkdir -p "$BASE_OUTPUT_DIR"

# ======== 4. 执行循环评估 ========
for MODEL_PATH in "${MODELS[@]}"; do
    # 只有当路径以 / 或 . 开头时，才检查本地目录是否存在
    if [[ "$MODEL_PATH" == /* ]] || [[ "$MODEL_PATH" == .* ]]; then
        if [ ! -d "$MODEL_PATH" ]; then
            echo "Warning: Local model path $MODEL_PATH not found, skipping..."
            continue
        fi
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

        # 1. 运行数据并行 + 张量并行推理 (DP + TP)
        PIDS=()
        NUM_REPLICAS=$((NUM_GPUS / TP_SIZE))
        for (( i=0; i<$NUM_REPLICAS; i++ )); do
            # 计算每张卡分配的 GPU ID (例如 TP_SIZE=2 时，i=0 用 0,1；i=1 用 2,3)
            DEVICES=""
            for (( j=0; j<$TP_SIZE; j++ )); do
                gpu_id=$((i * TP_SIZE + j))
                if [ -z "$DEVICES" ]; then
                    DEVICES="$gpu_id"
                else
                    DEVICES="$DEVICES,$gpu_id"
                fi
            done
            
            CUDA_VISIBLE_DEVICES=$DEVICES python eval.py \
              --model_name="$MODEL_PATH" \
              --datasets="$DATASET" \
              --split="$DS_SPLIT" \
              --output_dir="$CURRENT_OUTPUT_DIR" \
              --max_tokens=$MAX_TOKENS \
              --num_gpus=$TP_SIZE \
              --temperature=$TEMPERATURE \
              --top_p=$TOP_P \
              --top_k=$TOP_K \
              --num_generation=$NUM_GENERATION \
              --num_shards=$NUM_REPLICAS \
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