i2v_models=("LTX-2 Wan2.2")
robot_types=("dual_arm" "humanoid" "single_arm" "quad")
vlm_list=("gpt" "qwen")   # gpt / qwen

# You need to replace the api key with your own
gpt_api_key="sk-XXXXXXXX"
qwen_api_key="sk-XXXXXXXX"

for I2VMODEL_NAME in "${i2v_models[@]}"; do
  echo "======================================================="
  echo "Start evaluating: $I2VMODEL_NAME"
  echo "======================================================="

  for VLM_NAME in "${vlm_list[@]}"; do
    export VLM_NAME=$VLM_NAME

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

    for ROBOT_TYPE in "${robot_types[@]}"; do
      echo "------------------------------"
      echo "ROBOT TYPE: ${ROBOT_TYPE}，VLM=${VLM_NAME}"
      echo "------------------------------"

      # ================================
      # Step 1~3: VQA with gpt/qwen
      # ================================
      echo "Step 1: robot subject stability evaluation (${VLM_NAME})..."
      python eval/4_embodiments/1_robot_subject_stability.py \
        --model ${VLM_NAME} \
        --video_path data/${I2VMODEL_NAME}/${ROBOT_TYPE}/videos \
        --image_grid_path data/${I2VMODEL_NAME}/${ROBOT_TYPE}/image_grid_2frame \
        --output_path results/4_embodiments/${I2VMODEL_NAME}/${ROBOT_TYPE}/VQA/${VLM_NAME}/1_robot_subject_stability \
        --read_prompt_file data/prompts/${ROBOT_TYPE}_prompts.json \
        --api_key "$API_KEY" \
        --num_workers "$NUM_WORKER"

      echo "Step 2: physical plausibility evaluation (${VLM_NAME})..."
      python eval/4_embodiments/2_physical_plausibility.py \
        --model ${VLM_NAME} \
        --video_path data/${I2VMODEL_NAME}/${ROBOT_TYPE}/videos \
        --image_grid_path data/${I2VMODEL_NAME}/${ROBOT_TYPE}/image_grid_6frame \
        --output_path results/4_embodiments/${I2VMODEL_NAME}/${ROBOT_TYPE}/VQA/${VLM_NAME}/2_physical_plausibility \
        --read_prompt_file data/prompts/${ROBOT_TYPE}_prompts.json \
        --api_key "$API_KEY" \
        --num_workers "$NUM_WORKER"

      echo "Step 3: task adherence consistency evaluation (${VLM_NAME})..."
      python eval/4_embodiments/3_task_adherence_consistency.py \
        --model ${VLM_NAME} \
        --video_path data/${I2VMODEL_NAME}/${ROBOT_TYPE}/videos \
        --image_grid_path data/${I2VMODEL_NAME}/${ROBOT_TYPE}/image_grid_6frame \
        --output_path results/4_embodiments/${I2VMODEL_NAME}/${ROBOT_TYPE}/VQA/${VLM_NAME}/3_task_adherence_consistency \
        --read_prompt_file data/prompts/${ROBOT_TYPE}_prompts.json \
        --api_key "$API_KEY" \
        --num_workers "$NUM_WORKER"
    done
  done

  # =====================================================
  #   Step 4~7：Motion
  # =====================================================
  echo "🚀 Start evaluating Motion-related metrics"
  for ROBOT_TYPE in "${robot_types[@]}"; do
    echo "------------------------------"
    echo "ROBOT TYPE: ${ROBOT_TYPE}"
    echo "------------------------------"

    VIDEO_DIR="data/${I2VMODEL_NAME}/${ROBOT_TYPE}/videos"
    META_PATH="data/prompts/${ROBOT_TYPE}_prompts.json"
    META_INFO_PATH="./results/4_embodiments/${I2VMODEL_NAME}/${ROBOT_TYPE}/motion/results.json"

    echo "Step 4: Create meta info..."
    python eval/4_embodiments/4_create_meta_info.py -v $VIDEO_DIR -o $META_INFO_PATH -i $META_PATH

    echo "Step 5: Robotic PAS..."
    python eval/4_embodiments/5_motion_amplitude.py --meta_info_path $META_INFO_PATH \
        --target_type "robotic_manipulator" \
        --box_threshold 0.25 --text_threshold 0.20 --grid_size 30 --device cuda

    echo "Step 6: Motion Smoothness..."
    python eval/4_embodiments/6_motion_smoothness.py --meta_info_path $META_INFO_PATH

    echo "Step 7: Motion Total Score..."
    python eval/4_embodiments/7_motion_total_score.py --meta_info_path $META_INFO_PATH

    echo "Step 8: ${ROBOT_TYPE} Results..."
    python eval/4_embodiments/8_summarize_robot_results.py --i2v_model_name $I2VMODEL_NAME --robot_type $ROBOT_TYPE
  done
  
  echo "🎉 Finish $I2VMODEL_NAME Evaluation"
  echo "Step 9: $I2VMODEL_NAME Results..."
  python eval/4_embodiments/summarize_i2v_results.py --i2v_model_name $I2VMODEL_NAME
done

echo "🎉 Evaluation completed for all I2V models！"
python eval/4_embodiments/summary_scores.py --root_dir results/4_embodiments

