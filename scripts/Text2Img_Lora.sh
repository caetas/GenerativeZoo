export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./../../../../models/Text2Img_Lora/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"

cd ./../src/generativezoo/models/SD

accelerate launch --mixed_precision="fp16"  Text2Img_Lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_PATH \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue wings." \
  --seed=1337 \
  --image_column="image" \
  --caption_column="text"