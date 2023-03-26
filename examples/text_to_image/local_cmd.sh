# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export dataset_name="lambdalabs/pokemon-blip-captions"

# accelerate launch --gpu_ids '3,4' --num_processes 2 --num_machines 1 --mixed_precision="fp16"  train_text_to_image.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$dataset_name \
#   --use_ema \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=6 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=1 \
#   --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --output_dir="sd-pokemon-model" \
#   --cache_dir="/localscratch/renjie/huggingface"

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="./beihong/dataset"

accelerate launch --main_process_port 9527 --gpu_ids '2, 3' --num_processes 1 --num_machines 1 --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=6 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=2000 \
  --learning_rate=5e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="results/sd-beihong-model" \
  --checkpointing_steps 100

# array=(500 1000 1500 2000)

# for(( i=0;i<${#array[@]};i++)) do
# echo ${array[i]};
# CUDA_VISIBLE_DEVICES='6' python text2img.py --checkpoint ${array[i]}
# done;

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# CUDA_VISIBLE_DEVICES='0' python text2img.py --checkpoint 1000 --ema --pretrained_model_name_or_path=$MODEL_NAME --model_path ./results/sd-beihong-model/ --prompt_postfix " by Beihong Xu" --prompt_file prompt

# scp -r /egr/research-dselab/renjie3/renjie/diffusion/sd_finetune/diffusers/examples/text_to_image/results/sd-beihong-model renjie3@hpcc.msu.edu:/mnt/home/renjie3/Documents/unlearnable/diffusion/diffusers/examples/text_to_image/results
