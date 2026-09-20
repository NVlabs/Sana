# GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment

This repository contains code for the paper [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513) by Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt.

TLDR: We demonstrate the advantages of evaluating text-to-image models using existing object detection methods, to produce a fine-grained instance-level analysis of compositional capabilities.

### Abstract

*Recent breakthroughs in diffusion models, multimodal pretraining, and efficient finetuning have led to an explosion of text-to-image generative models.
Given human evaluation is expensive and difficult to scale, automated methods are critical for evaluating the increasingly large number of new models.
However, most current automated evaluation metrics like FID or CLIPScore only offer a holistic measure of image quality or image-text alignment, and are unsuited for fine-grained or instance-level analysis.
In this paper, we introduce GenEval, an object-focused framework to evaluate compositional image properties such as object co-occurrence, position, count, and color.
We show that current object detection models can be leveraged to evaluate text-to-image models on a variety of generation tasks with strong human agreement, and that other discriminative vision models can be linked to this pipeline to further verify properties like object color.
We then evaluate several open-source text-to-image models and analyze their relative generative capabilities on our benchmark.
We find that recent models demonstrate significant improvement on these tasks, though they are still lacking in complex capabilities such as spatial relations and attribute binding.
Finally, we demonstrate how GenEval might be used to help discover existing failure modes, in order to inform development of the next generation of text-to-image models.*

### Summary figure

<p align="center">
    <img src="images/geneval_figure_1.png" alt="figure1"/>
</p>

### Main results

| Model | Overall | <span style="font-weight:normal">Single object</span> | <span style="font-weight:normal">Two object</span> | <span style="font-weight:normal">Counting</span> | <span style="font-weight:normal">Colors</span> | <span style="font-weight:normal">Position</span> | <span style="font-weight:normal">Color attribution</span> |
| ----- | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| CLIP retrieval (baseline) | **0.35** | 0.89 | 0.22 | 0.37 | 0.62 | 0.03 | 0.00 |
minDALL-E | **0.23** | 0.73 | 0.11 | 0.12 | 0.37 | 0.02 | 0.01 |
Stable Diffusion v1.5 | **0.43** | 0.97 | 0.38 | 0.35 | 0.76 | 0.04 | 0.06 |
Stable Diffusion v2.1 | **0.50** | 0.98 | 0.51 | 0.44 | 0.85 | 0.07 | 0.17 |
Stable Diffusion XL | **0.55** | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 |
IF-XL | **0.61** | 0.97 | 0.74 | 0.66 | 0.81 | 0.13 | 0.35 |

## Code

### Setup

Install the dependencies, including `mmdet`, and download the Mask2Former object detector:

```bash
git clone https://github.com/djghosh13/geneval.git
cd geneval
conda env create -f environment.yml
conda activate geneval
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

The original GenEval prompts from the paper are already in `prompts/`, but you can sample new prompts with different random seeds using

```bash
python prompts/create_prompts.py --seed <SEED> -n <NUM_PROMPTS> -o "<PROMPT_FOLDER>/"
```

### Image generation

Sample image generation code for Stable Diffusion models is given in `generation/diffusers_generate.py`. Run

```bash
python generation/diffusers_generate.py \
    "<PROMPT_FOLDER>/evaluation_metadata.jsonl" \
    --model "runwayml/stable-diffusion-v1-5" \
    --outdir "<IMAGE_FOLDER>"
```

to generate 4 images per prompt using Stable Diffusion v1.5 and save in `<IMAGE_FOLDER>`.

The generated format should be

```
<IMAGE_FOLDER>/
    00000/
        metadata.jsonl
        grid.png
        samples/
            0000.png
            0001.png
            0002.png
            0003.png
    00001/
        ...
```

where `metadata.jsonl` contains the `N`-th line from `evaluation_metadata.jsonl`. `grid.png` is optional here.

### Evaluation

```bash
python evaluation/evaluate_images.py \
    "<IMAGE_FOLDER>" \
    --outfile "<RESULTS_FOLDER>/results.jsonl" \
    --model-path "<OBJECT_DETECTOR_FOLDER>"
```

This will result in a JSONL file with each line corresponding to an image. In particular, each line has a `correct` key and a `reason` key specifying whether the generated image was deemed correct and, if applicable, why it was marked incorrect. You can run

```bash
python evaluation/summary_scores.py "<RESULTS_FOLDER>/results.jsonl"
```

to get the score across each task, and the overall GenEval score.
