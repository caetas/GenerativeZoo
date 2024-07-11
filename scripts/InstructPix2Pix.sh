export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="./../../../../models/InstructPix2Pix/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"

cd ./../src/generativezoo/models/SD

accelerate launch --mixed_precision="fp16" InstructPix2Pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_PATH \
    --output_dir=$OUTPUT_DIR \
    --resolution=512 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --validation_image "./../../../../data/processed/pokemons/conditioning_images/0003_mask.png" \
    --validation_prompt "red circle pokemon with white dots" \
    --seed=42 \
    --report_to=wandb \
    --original_image_colum="original_image" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt"