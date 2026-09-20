[![PyPI](https://img.shields.io/pypi/v/pytorch-fid.svg)](https://pypi.org/project/pytorch-fid/)

# FID score for PyTorch

This is a port of the official implementation of [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500) to PyTorch.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) for the original implementation using Tensorflow.

FID is a measure of similarity between two datasets of images.
It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network.

Further insights and an independent evaluation of the FID score can be found in [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337).

The weights and the model are exactly the same as in [the official Tensorflow implementation](https://github.com/bioinf-jku/TTUR), and were tested to give very similar results (e.g. `.08` absolute error and `0.0009` relative error on LSUN, using ProGAN generated images). However, due to differences in the image interpolation implementation and library backends, FID results still differ slightly from the original implementation. So if you report FID scores in your paper, and you want them to be *exactly comparable* to FID scores reported in other papers, you should consider using [the official Tensorflow implementation](https://github.com/bioinf-jku/TTUR).

## Installation

Install from [pip](https://pypi.org/project/pytorch-fid/):

```
pip install pytorch-fid
```

Requirements:

- python3
- pytorch
- torchvision
- pillow
- numpy
- scipy

## Usage

To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:

```
python -m pytorch_fid path/to/dataset1 path/to/dataset2
```

To run the evaluation on GPU, use the flag `--device cuda:N`, where `N` is the index of the GPU to use.

### Using different layers for feature maps

In difference to the official implementation, you can choose to use a different feature layer of the Inception network instead of the default `pool3` layer.
As the lower layer features still have spatial extent, the features are first global average pooled to a vector before estimating mean and covariance.

This might be useful if the datasets you want to compare have less than the otherwise required 2048 images.
Note that this changes the magnitude of the FID score and you can not compare them against scores calculated on another dimensionality.
The resulting scores might also no longer correlate with visual quality.

You can select the dimensionality of features to use with the flag `--dims N`, where N is the dimensionality of features.
The choices are:

- 64: first max pooling features
- 192: second max pooling features
- 768: pre-aux classifier features
- 2048: final average pooling features (this is the default)

## Generating a compatible `.npz` archive from a dataset

A frequent use case will be to compare multiple models against an original dataset.
To save training multiple times on the original dataset, there is also the ability to generate a compatible `.npz` archive from a dataset. This is done using any combination of the previously mentioned arguments with the addition of the `--save-stats` flag. For example:

```
python -m pytorch_fid --save-stats path/to/dataset path/to/outputfile
```

The output file may then be used in place of the path to the original dataset for further comparisons.

## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:

```
@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.3.0},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```

## License

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).
