# Stable Diffusion 2.1 Text-to-Image with LoRA

## Prepare the Dataset

If you want to train the model in a custom dataset, two elements must be provided: the `images` and the `prompts`. The images must be provided in a folder organized as follows:

├── dataset
│   ├── images
│   │   ├── <name0>.png
│   │   ├── <name1>.png
│   │   ├── ...
│   ├── train.jsonl

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

A script file is provided with the commands required to train the model on a custom dataset. Several parameters should be configured, mainly:

```sh
export OUTPUT_DIR="./../../../../models/Text2Img_Lora/pokemons"
export DATASET_PATH="./../../../../data/processed/pokemons"
```

## Inference