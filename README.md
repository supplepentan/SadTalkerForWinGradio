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
python scripts/download_models.py
```

## Run the Gradio

```bash
python launcher.py
```

Access to http://127.0.0.1:7860 .

## Acknowledgements

This code is built on [SadTalker](https://github.com/OpenTalker/SadTalker), thank for the authors for sharing their codes.
