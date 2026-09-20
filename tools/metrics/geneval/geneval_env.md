The installation process refers to https://github.com/djghosh13/geneval/issues/12 Thanks for the community!

# Cloning the repository

```bash
git clone https://github.com/djghosh13/geneval.git

cd geneval
conda create -n geneval python=3.8.10 -y
conda activate geneval
```

# Installing dependencies

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
python -m pip install lightning
pip install diffusers["torch"] transformers
pip install tomli
pip install platformdirs
pip install --upgrade setuptools
```

# mmengine and mmcv dependency installation

```bash
mim install mmengine mmcv-full==1.7.2
```

# mmdet installation

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

<!-- conda create -n geneval python==3.9 -->

<!--
conda install -c nvidia cuda-toolkit -y
./evaluation/download_models.sh output/

# install mmdetection

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .

pip install einops
pip install accelerate==0.15.0
pip install torchmetrics==0.6.0
pip install transformers==4.48.0
pip install pandas
pip install open-clip-torch
pip install clip_benchmark
pip install huggingface_hub==0.24.5
pip install pytorch-lightning==1.4.2

pip install openmim
mim install mmcv-full==1.7.1 -->
