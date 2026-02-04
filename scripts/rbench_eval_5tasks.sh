i2v_models=("LTX-2 Wan2.2")
task_cfg=(
  "common_manipulation:6"
  "long-horizon_planning:6"
  "multi-entity_collaboration:6"
  "spatial_relationship:3"
  "visual_reasoning:6"
)
vlm_list=("gpt" "qwen")   # gpt / qwen

# You need to replace the api key with your own
gpt_api_key="sk-XXXXXXXXXX"
qwen_api_key="sk-XXXXXXXXXX"

for I2VMODEL_NAME in "${i2v_models[@]}"; do
  echo "======================================"
  echo "Start evaluating: $I2VMODEL_NAME"
  echo "======================================"

  for VLM_NAME in "${vlm_list[@]}"; do
    export VLM_NAME
    if [[ "$VLM_NAME" == "gpt" ]]; then
        export API_KEY="$gpt_api_key"
        export NUM_WORKER=8

    elif [[ "$VLM_NAME" == "qwen" ]]; then
        export API_KEY="$qwen_api_key"
        export NUM_WORKER=1
    else
        echo "❌ Unknown VLM_NAME: $VLM_NAME"
        exit 1
    fi    

    for item in "${task_cfg[@]}"; do
      TASK_TYPE="${item%%:*}"
      GRID_NUM="${item##*:}"
      GRID_PATH="image_grid_${GRID_NUM}frame"

      echo "------------------------------"
      echo "TASK TYPE: ${TASK_TYPE}, VLM=${VLM_NAME}"
      echo "------------------------------"

      python eval/5_tasks/${TASK_TYPE}.py \
        --model ${VLM_NAME} \
        --video_path data/${I2VMODEL_NAME}/${TASK_TYPE}/videos \
        --image_grid_path data/${I2VMODEL_NAME}/${TASK_TYPE}/${GRID_PATH} \
        --output_path results/5_tasks/${I2VMODEL_NAME}/${TASK_TYPE}/${VLM_NAME} \
        --read_prompt_file data/prompts/${TASK_TYPE}_prompts.json \
        --api_key "$API_KEY" \
        --num_workers "$NUM_WORKER"
      echo
    done
  done
done
echo "🎉 Evaluation completed for all I2V models！"
python eval/5_tasks/summary_scores.py --root_dir results/5_tasks
