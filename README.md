# SadTalker for Win

## Introduction

This is Windows model of SadTalker using Gradio.

Original: [SadTalker](https://github.com/OpenTalker/SadTalker)

### Dependencies

- OS: Windows
- nvidia:
  - cuda: 11.7
- python 3.10.5 (using Pyenv)

## Install

## Clone from GitHub

Clone from GitHub, and move to the folder " SadTalker ".

```bash
git clone https://github.com/supplepentan/SadTalkerForWinGradio
cd SadTalkerForWinGradio
```

## Virtual environment

Virtual environment with Python-version 3.10.5 using Pyenv.

```bash
pyenv local 3.10.5
python -m venv venv
venv/scripts/activate
```

## Libraries

Install libraries using " reqruirements.txt "

```bash
python -m pip install -r requirements.txt
```

### Pytorch

Uninstall unnecessary installed Pytorch, and newly install Pytorch (CUDA11.7 compatible model).

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

### Download Models

```bash
python utils/download_models.py
```

## Run the Gradio

```bash
python app.py
```

Access to http://127.0.0.1:7860 .

## Acknowledgements

This code is built on [SadTalker](https://github.com/OpenTalker/SadTalker), thank for the authors for sharing their codes.

## Appendix: Core Dependencies

numpy==1.23.4
face_alignment==1.3.5
imageio==2.19.3
imageio-ffmpeg==0.4.7
librosa==0.9.2
numba
resampy==0.3.1
pydub==0.25.1
scipy==1.10.1
kornia==0.6.8
tqdm
yacs==0.1.8
pyyaml
joblib==1.1.0
scikit-image==0.19.3
basicsr==1.4.2
facexlib==0.3.0
gradio==5.35.0
gfpgan
av
safetensors
requests
