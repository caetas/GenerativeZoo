export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./../../../../models/Text2Img_Controlnet/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"

cd ./../src/generativezoo/models/SD

accelerate launch --mixed_precision="fp16"  Text2Img_Controlnet.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATASET_PATH \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=10000 \
 --checkpointing_steps=20 \
 --validation_image "./../../../../data/processed/pokemons/conditioning_images/0003_mask.png" \
 --validation_prompt "red circle pokemon with white dots" \
 --train_batch_size=1 \
 --report_to=wandb \
 --image_column="image" \
 --caption_column="text" \
 --conditioning_image_column='conditioning_image' \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam