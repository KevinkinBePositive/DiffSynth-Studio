modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-I2V-A14B/*" --local_dir ./data/diffsynth_example_dataset

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

COMMON_MODEL_CONFIGS="Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"
DMD2_DIT_CONFIGS="Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors"
AUX_MODEL_CONFIGS="Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"
DMD2_CACHE_PATH="./models/train/Wan2.2-I2V-A14B_dmd2_lowvram_cache"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 21 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "${COMMON_MODEL_CONFIGS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${DMD2_CACHE_PATH}" \
  --task "dmd2_distill:data_process" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --offload_models "${DMD2_DIT_CONFIGS}" \
  --fp8_models "${AUX_MODEL_CONFIGS}" \
  --use_gradient_checkpointing_offload

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DMD2_CACHE_PATH}" \
  --height 480 \
  --width 832 \
  --num_frames 21 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "${DMD2_DIT_CONFIGS}" \
  --dmd2_teacher_model_id_with_origin_paths "${DMD2_DIT_CONFIGS}" \
  --dmd2_guidance_model_id_with_origin_paths "${DMD2_DIT_CONFIGS}" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-I2V-A14B_high_noise_dmd2_8step_lora" \
  --task "dmd2_distill:train" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --dmd2_guidance_lora_rank 32 \
  --dmd2_guidance_lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --extra_inputs "input_image" \
  --dmd2_teacher_fp8_models "" \
  --dmd2_teacher_offload_models "${DMD2_DIT_CONFIGS}" \
  --use_gradient_checkpointing_offload \
  --dmd2_num_inference_steps 8 \
  --dmd2_student_cfg_scale 1.0 \
  --dmd2_real_guidance_scale 4.0 \
  --dmd2_fake_guidance_scale 1.0 \
  --dmd2_dm_loss_weight 1.0 \
  --dmd2_fake_loss_weight 1.0 \
  --dmd2_min_step_percent 0 \
  --dmd2_max_step_percent 0.358 \
  --dmd2_switch_DiT_boundary 0.9 \
  --dmd2_sigma_shift 5.0
# DMD2 score boundary corresponds to high-noise timesteps [900, 1000].

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DMD2_CACHE_PATH}" \
  --height 480 \
  --width 832 \
  --num_frames 21 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "${DMD2_DIT_CONFIGS}" \
  --dmd2_teacher_model_id_with_origin_paths "${DMD2_DIT_CONFIGS}" \
  --dmd2_guidance_model_id_with_origin_paths "${DMD2_DIT_CONFIGS}" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit2." \
  --output_path "./models/train/Wan2.2-I2V-A14B_low_noise_dmd2_8step_lora" \
  --task "dmd2_distill:train" \
  --lora_base_model "dit2" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --dmd2_guidance_lora_rank 32 \
  --dmd2_guidance_lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --extra_inputs "input_image" \
  --dmd2_teacher_fp8_models "" \
  --dmd2_teacher_offload_models "${DMD2_DIT_CONFIGS}" \
  --use_gradient_checkpointing_offload \
  --dmd2_num_inference_steps 8 \
  --dmd2_student_cfg_scale 1.0 \
  --dmd2_real_guidance_scale 4.0 \
  --dmd2_fake_guidance_scale 1.0 \
  --dmd2_dm_loss_weight 1.0 \
  --dmd2_fake_loss_weight 1.0 \
  --dmd2_min_step_percent 0.358 \
  --dmd2_max_step_percent 1 \
  --dmd2_switch_DiT_boundary 0.9 \
  --dmd2_sigma_shift 5.0
# DMD2 score boundary corresponds to low-noise timesteps [0, 900).
