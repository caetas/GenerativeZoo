# Stable Diffusion 2.1 Text-to-Image with LoRA

## Prepare the Dataset

If you want to train the model in a custom dataset, two elements must be provided: the `images` and the `prompts`. The images must be provided in a folder organized as follows:

```bash
├── dataset
│   ├── images
│   │   ├── <name0>
│   │   ├── <name1>
│   │   ├── ...
│   ├── train.jsonl
```

The file `train.jsonl` contains the prompts associated to each image and should be structured like the following example:

```json
{"text": "a drawing of a green pokemon with red eyes", "image": "./../../../../data/processed/pokemons/images/0000.png"}
{"text": "a green and yellow toy with a red nose", "image": "./../../../../data/processed/pokemons/images/0001.png"}
...
```

## Accelerator Config

Please configure the accelerator to match your system requirements by running:

```bash
accelerate config
```

## Train the model

A [`script file`](./../scripts/Text2Img_Lora.sh) is provided with the commands required to train the model on a custom dataset. Several parameters should be configured, mainly:

```sh
export OUTPUT_DIR="./../../../../models/Text2Img_Lora/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"
```

Several parameters in the training command can be tuned by the user, particularly the `--validation_prompt` which should reflect the use case of this training.

```sh
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
```

## Inference

A Python script is provided to use the trained LoRA adapters:

```bash
python Text2Img_LoRA.py --lora_model_path ./../../models/Text2Img_Lora/pokemons/checkpoint-60 
```