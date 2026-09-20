FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

COPY pyproject.toml pyproject.toml
COPY diffusion diffusion
COPY configs configs
COPY sana sana
COPY app app
COPY tools tools

COPY environment_setup.sh environment_setup.sh
RUN ./environment_setup.sh

CMD ["python", "-u", "-W", "ignore", "app/app_sana.py", "--share", "--config=configs/sana_config/1024ms/Sana_1600M_img1024.yaml", "--model_path=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth"]
