# ControlNet

## Prepare the Dataset

If you want to train the model in a custom dataset, three elements must be provided: the `images`, the `conditioning images` and the `prompts`. The images must be provided in a folder organized as follows:

```bash
├── dataset
│   ├── images
│   │   ├── <name0>
│   │   ├── <name1>
│   │   ├── ...
│   ├── conditioning_images
│   │   ├── <cond_name0>
│   │   ├── <cond_name1>
│   │   ├── ...
│   ├── train.jsonl
```

The file `train.jsonl` contains the prompts associated to each image and conditioning image and should be structured like the following example:

```json
{"text": "a drawing of a green pokemon with red eyes", "image": "./../../../../data/processed/pokemons/images/0000.png", "conditioning_image": "./../../../../data/processed/pokemons/conditioning_images/0000_mask.png"}
{"text": "a green and yellow toy with a red nose", "image": "./../../../../data/processed/pokemons/images/0001.png", "conditioning_image": "./../../../../data/processed/pokemons/conditioning_images/0001_mask.png"}
...
```

## Accelerator Config

Please configure the accelerator to match your system requirements by running:

```bash
accelerate config
```

## Train the model

A [`script file`](./../scripts/Text2Img_Controlnet.sh) is provided with the commands required to train the model on a custom dataset. Several parameters should be configured, mainly:

```sh
export OUTPUT_DIR="./../../../../models/Text2Img_Controlnet/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"
```

Several parameters in the training command can be tuned by the user, particularly the `--validation_prompt` and `--validation_image` which should reflect the use case of this training.

```sh
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
python Text2Img_ControlNet.py --cnet_model_path ./../../models/Text2Img_ControlNet/pokemons/checkpoint-200/controlnet --cond_image_path ./../../data/processed/pokemons/conditioning_images/0003_mask.png 
```