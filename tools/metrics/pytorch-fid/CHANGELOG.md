# Changelog

## [0.3.0] - 2023-01-05

### Added

- Add argument `--save-stats` allowing to compute dataset statistics and save them as an `.npz` file ([#80](https://github.com/mseitzer/pytorch-fid/pull/80)). The `.npz` file can be used in subsequent FID computations instead of recomputing the dataset statistics. This option can be used in the following way: `python -m pytorch_fid --save-stats path/to/dataset path/to/outputfile`.

### Fixed

- Do not use `os.sched_getaffinity` to get number of available CPUs on Windows, as it is not available there ([232b3b14](https://github.com/mseitzer/pytorch-fid/commit/232b3b1468800102fcceaf6f2bb8977811fc991a), [#84](https://github.com/mseitzer/pytorch-fid/issues/84)).
- Do not use Inception model argument `pretrained`, as it was deprecated in torchvision 0.13 ([#88](https://github.com/mseitzer/pytorch-fid/pull/88)).

## [0.2.1] - 2021-10-10

### Added

- Add argument `--num-workers` to select number of dataloader processes ([#66](https://github.com/mseitzer/pytorch-fid/pull/66)). Defaults to 8 or the number of available CPUs if less than 8 CPUs are available.

### Fixed

- Fixed package setup to work under Windows ([#55](https://github.com/mseitzer/pytorch-fid/pull/55), [#72](https://github.com/mseitzer/pytorch-fid/issues/72))

## [0.2.0] - 2020-11-30

### Added

- Load images using a Pytorch dataloader, which should result in a speed-up. ([#47](https://github.com/mseitzer/pytorch-fid/pull/47))
- Support more image extensions ([#53](https://github.com/mseitzer/pytorch-fid/pull/53))
- Improve tooling by setting up Nox, add linting and test support ([#52](https://github.com/mseitzer/pytorch-fid/pull/52))
- Add some unit tests

## [0.1.1] - 2020-08-16

### Fixed

- Fixed software license string in `setup.py`

## [0.1.0] - 2020-08-16

Initial release as a pypi package. Use `pip install pytorch-fid` to install.
