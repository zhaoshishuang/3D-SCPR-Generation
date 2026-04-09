# 3D-SCPR-Generation

This repository is established for the paper `Clinical Risk-Aware Multi-Level Grading for Coronary
Artery Stenosis through Curved Feature Reconstruction` and provides an implementation of the 3D SCPR image generation method. The project focuses on 3D centerline smoothing, resampling, and volume reconstruction based on local orthogonal coordinate systems, aiming to generate straightened 3D SCPR representations from original 3D images.

## Repository Structure

- `main.py`: A minimal example entry point demonstrating how to call the centerline smoothing and 3D SCPR generation functions.
- `utils.py`: Core algorithm implementations, including cumulative curve distance computation, vector normalization, local axis construction, centerline smoothing and resampling, and straightened 3D SCPR generation.

## Usage

### Requirements

The project depends on the following Python libraries:

- `numpy`
- `scipy`
- `scikit-image`

They can be installed as needed:

```bash
pip install numpy scipy scikit-image
```

### Run Example

```bash
python main.py
```

`main.py` demonstrates a minimal workflow:

- Construct sample centerline points and 3D volume data.
- Smooth and resample the centerline.
- Generate the 3D SCPR result based on the local coordinate system along the centerline.

## 📎 Citation

```bibtex

```