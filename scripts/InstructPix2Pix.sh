export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./../../../../models/InstructPix2Pix/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"

cd ./../src/generativezoo/models/SD

accelerate launch --mixed_precision="fp16" train_InstructPix2Pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
    --validation_prompt="make the mountains snowy" \
    --seed=42 \
    --report_to=wandb \