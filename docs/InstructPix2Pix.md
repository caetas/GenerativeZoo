# InstructPix2Pix

## Prepare the Dataset

If you want to train the model in a custom dataset, three elements must be provided: the `original images`, the `edited images` and the `edit prompts`. The images must be provided in a folder organized as follows:

```bash
├── dataset
│   ├── original_images
│   │   ├── <name0>
│   │   ├── <name1>
│   │   ├── ...
│   ├── edited_images
│   │   ├── <edit_name0>
│   │   ├── <edit_name1>
│   │   ├── ...
│   ├── train.jsonl
```

The file `train.jsonl` contains the edit prompts associated to each original image and edited image and should be structured like the following example:

```json
{"edit_prompt": "a drawing of a green pokemon with red eyes", "original_image": "./../../../../data/processed/pokemons/images/0000.png", "edited_image": "./../../../../data/processed/pokemons/conditioning_images/0000_mask.png"}
{"edit_prompt": "a green and yellow toy with a red nose", "original_image": "./../../../../data/processed/pokemons/images/0001.png", "edited_image": "./../../../../data/processed/pokemons/conditioning_images/0001_mask.png"}
...
```

## Accelerator Config

Please configure the accelerator to match your system requirements by running:

```bash
accelerate config
```

## Train the model

A [`script file`](./../scripts/InstructPix2Pix.sh) is provided with the commands required to train the model on a custom dataset. Several parameters should be configured, mainly:

```sh
export OUTPUT_DIR="./../../../../models/InstructPix2Pix/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"
```

Several parameters in the training command can be tuned by the user, particularly the `--validation_prompt` and `--validation_image` which should reflect the use case of this training.

```sh
accelerate launch --mixed_precision="fp16" train_InstructPix2Pix.py \
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
```

For GPUs with less VRAM, you might consider using some of the following options to reduce memory usage:

```sh
 --use_8bit_adam \
 --gradient_checkpointing \
 --set_grads_to_none \
```

## Inference

A Python script is provided to use the trained ControlNet adapters:

```bash
python InstructPix2Pix.py  --pix2pix_model ./../../models/InstructPix2Pix/checkpoint-20/unet --image_path ./../../data/processed/pokemons/conditioning_images/0003_mask.png
```