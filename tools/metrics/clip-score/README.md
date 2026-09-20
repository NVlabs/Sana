# CLIP Score for PyTorch

[![PyPI](https://img.shields.io/pypi/v/clip-score.svg)](https://pypi.org/project/clip-score/)

This repository provides a batch-wise quick processing for calculating CLIP scores. It uses the pretrained CLIP model to measure the cosine similarity between two modalities. The project structure is adapted from [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [CLIP](https://github.com/openai/CLIP).

## Installation

Requirements:

- Install PyTorch:
  ```
  pip install torch  # Choose a version that suits your GPU
  ```
- Install CLIP:
  ```
  pip install git+https://github.com/openai/CLIP.git
  ```
- Install clip-score from [PyPI](https://pypi.org/project/clip-score/):
  ```
  pip install clip-score
  ```

## Data Input Specifications

This project is designed to process paired images and text files, and therefore requires two directories: one for images and one for text files.

### Image Files

All images should be stored in a single directory. The image files can be in either `.png` or `.jpg` format.

### Text Files

All text data should be contained in plain text files in a separate directory. These text files should have the extension `.txt`.

### File Number and Naming

The number of files in the image directory should be exactly equal to the number of files in the text directory. Additionally, the files in the image directory and text directory should be paired by file name. For instance, if there is a `cat.png` in the image directory, there should be a corresponding `cat.txt` in the text directory.

### Directory Structure Example

Below is an example of the expected directory structure:

```plaintext
├── path/to/image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
└── path/to/text
    ├── cat.txt
    ├── dog.txt
    └── bird.txt
```

In this example, `cat.png` is paired with `cat.txt`, `dog.png` is paired with `dog.txt`, and `bird.jpg` is paired with `bird.txt`.

Please adhere to the specified structure to ensure correct operation of the program. If there are any questions or issues, feel free to raise an issue here on GitHub.

## Usage

To compute the CLIP score between images and texts, make sure that the image and text data are contained in two separate folders, and each sample has the same name in both modalities. Run the following command:

```
python -m clip_score path/to/image path/to/text
```

If GPU is available, the project is set to run automatically on a GPU by default. If you want to specify a particular GPU, you can use the `--device cuda:N` flag when running the script, where `N` is the index of the GPU you wish to use. In case you want to run the program on a CPU instead, you can specify this by using the `--device cpu` flag.

## Computing CLIP Score within the Same Modality

If you want to calculate the CLIP score within the same modality (e.g., image-image or text-text), follow the same folder structure as mentioned above. Additionally, specify the preferred modalities using the `--real_flag` and `--fake_flag` options. By default, `--real_flag=img` and `--fake_flag=txt`. Examples:

```
python -m clip_score path/to/imageA path/to/imageB --real_flag img --fake_flag img
python -m clip_score path/to/textA path/to/textB --real_flag txt --fake_flag txt
```

## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:

```
@misc{taited2023CLIPScore,
  author={SUN Zhengwentai},
  title={{clip-score: CLIP Score for PyTorch}},
  month={March},
  year={2023},
  note={Version 0.1.1},
  howpublished={\url{https://github.com/taited/clip-score}},
}
```

## License

This implementation is licensed under the Apache License 2.0.

The project structure is adapted from [mseitzer's pytorch-fid](https://github.com/mseitzer/pytorch-fid) project. The CLIP model is adapted from [OpenAI's CLIP](https://github.com/openai/CLIP).

The CLIP Score was introduced in OpenAI's [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
