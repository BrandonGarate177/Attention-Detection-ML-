# Attention Detection

Real-time attention detection using face orientation, eye state, and head pose. Built with OpenCV, DeepFace, and a CNN-based classifier trained on labeled webcam data.

![Attention detection sample 1](https://github.com/user-attachments/assets/d128a5ac-6119-4525-a65f-1f7c87f61ef9)
![Attention detection sample 2](https://github.com/user-attachments/assets/33832f87-4084-4dfc-bcd7-76f2fc51ed21)

## Highlights
- Multi-signal attention scoring: facial orientation, eye tracking, and head pose.
- Custom dataset collection workflow with keyboard labeling.
- CNN training pipeline with transfer learning backbones (MobileNet, ResNet).
- Deployable model artifact (`.h5`) for downstream applications.

## How It Works
1. Collect labeled frames (Attentive vs. Distracted) from webcam.
2. Train a CNN on the collected dataset.
3. Run real-time inference on live video.

## Quick Start
The commands below are a template. Adjust paths and environment as needed.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Collect training data
python fire.py

# 4) Train the model
python train_model.py
```

## Data Collection
`fire.py` captures frames and labels them via keybinds:
- `a` = attentive
- `b` = distracted

Update the output directories at the top of `fire.py` to save into the desired folders.

## Training
`train_model.py` trains a TensorFlow model and saves an `.h5` file.
Expected dataset layout:

```
dataset/
  train/
  validation/
  test/
```

Note: training data and model artifacts are gitignored.

## Tech Stack
- OpenCV for video capture and preprocessing
- DeepFace for facial features
- TensorFlow/Keras for model training
- MobileNet/ResNet backbones for transfer learning

## Project Status
- macOS support is currently limited due to library compatibility on some CPUs.
- Some TensorFlow/Keras imports can fail in certain virtual environments.

## Roadmap
- Improve macOS compatibility.
- Add a lightweight on-device inference option.
- Expand dataset tooling for balanced sampling and augmentation.

## Goal
Build a robust attention detection model that can be deployed across learning, productivity, and safety applications.
