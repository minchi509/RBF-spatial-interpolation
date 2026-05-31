# Spatial Data Interpolation and Image Reconstruction using Radial Basis Functions (RBF)

A high-performance Python implementation of Radial Basis Function (RBF) interpolation for 3D spatial meteorological data and 2D image super-resolution. This project was developed as a Numerical Methods core assignment at Ho Chi Minh City University of Technology (HCMUT).

## 🚀 Features

- **Robust RBF Engine**: Built from scratch using `NumPy` and `SciPy`, supporting multiple kernel functions.
- **Problem 1 - Spatial Weather Interpolation**: Predicts maximum temperature ($T_{MAX}$) based on 3D coordinates (Longitude, Latitude, Elevation) from real-world Texas meteorological data. Generates 2D heatmaps and 3D terrain models.
- **Problem 2 - Image Super-Resolution**: Reconstructs and resizes multi-channel (RGB) images utilizing RBF interpolation, compared directly against standard Bicubic Spline.
- **Comprehensive Evaluation**: Automated calculation of standard error metrics: MAE, RMSE, and $R^2$ score.

---

## 📐 Mathematical Formulation

The interpolation model solves a block matrix system to find the blending weights $w$ and polynomial coefficients $v$:

$$\begin{bmatrix} \Phi & P \\ P^T & 0 \end{bmatrix} \begin{bmatrix} w \\ v \end{bmatrix} = \begin{bmatrix} y \\ 0 \end{bmatrix}$$

### Supported RBF Kernels:
- **Thin Plate Spline (TPS)**: $\phi(r) = r^2 \ln(r)$
- **Gaussian**: $\phi(r) = \exp(-(\epsilon r)^2)$
- **Multiquadric**: $\phi(r) = \sqrt{(\epsilon r)^2 + 1}$

---

## 📁 Project Structure

- `src/rbf_interpolator.py`: Core mathematical class handling distance matrices and matrix algebra (`scipy.linalg.solve`).
- `src/weather_interpolation.py`: Main execution script for spatial data processing, error metric logging, and 3D plotting.
- `src/image_resizer.py`: Pixel-wise matrix processing block dealing with multi-channel image interpolation.
- `data/`: Contains sample weather datasets (`.csv`) and testing templates.
- `results/`: Output visualization artifacts (Heatmaps, Comparison plots).

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.10+
- **Libraries**: `numpy`, `scipy`, `matplotlib`, `pandas`, `opencv-python`

### Installation
```bash
pip install -r requirements.txt
