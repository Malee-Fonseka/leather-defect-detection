# Leather Defect Detection

Mini-project for **EE7204/EC7205 — Image Processing and Computer Vision**, Department of Electrical and Information Engineering, University of Ruhuna.

The project detects and segments manufacturing defects (color stains, cuts, folds, glue marks, pokes) in leather images, with a focus on robustness under **low-light conditions**. It combines a diffusion-based synthetic data generator with a deep-learning segmentation model, and also includes a classical (non-ML) computer-vision baseline and a live-detection desktop GUI.

**Team:** Fonseka C.M. · Gunarathne A.N.G.T.N.B. · Piyumantha H.D.B.A.A. (Akila) · Rajapakshe R.M.M.L.D.

## Overview

Industrial leather inspection is often carried out under poor or inconsistent lighting, which degrades the accuracy of automated defect detectors. This project addresses that gap with the following pipeline:

1. **Simulate low-light conditions** on the [MVTec AD Leather](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (gamma darkening, Gaussian noise, vignetting, uneven illumination).
2. **Generate synthetic low-light defective samples** with a DDPM (Denoising Diffusion Probabilistic Model) to balance the class distribution, since real defective samples are scarce.
3. **Segment defects** pixel-wise with an Attention U-Net (ResNet50 encoder, Squeeze-Excite blocks, deep supervision) trained on both normal and low-light images, with CLAHE contrast enhancement as a preprocessing step.
4. **Quantify defect severity** by converting predicted masks into damaged-area estimates via the area calculator.
5. **Run live detection** on webcam, image or video input through the PyQt5 GUI (see [Live detection GUI](#live-detection-gui)).

A classical image-processing baseline (no deep learning) is also implemented for comparison, using reference-based anomaly detection, Gabor texture filtering, and multi-scale difference maps.

## Repository structure

| File | Description |
|---|---|
| [notebooks/data_pipeline.ipynb](notebooks/data_pipeline.ipynb) | Downloads/explores the MVTec AD Leather subset, applies low-light simulation, builds the preprocessing pipeline (resize, normalization, paired augmentation), creates train/val/test splits, and exposes a `tf.data.Dataset` factory. Verified with PSNR/SSIM. |
| [notebooks/Synthetic_image_generation.ipynb](notebooks/Synthetic_image_generation.ipynb) | Trains a DDPM (U-Net noise predictor, linear noise schedule, T=1000) on low-light defective images and samples new synthetic images to augment the defective class. Evaluated with PSNR/SSIM, saved to `synthetic_dataset/`. |
| [Leather_unet_multiclass_segmentation_tf.ipynb](Leather_unet_multiclass_segmentation_tf.ipynb) | **Main segmentation notebook.** Attention U-Net with a pretrained ResNet50 encoder, Squeeze-Excite blocks and deep supervision; weighted Focal + Dice loss with sqrt-inverse-frequency class weights; CLAHE preprocessing; full evaluation (per-class IoU/Dice, confusion matrix, failure-case analysis). |
| [Leather_unet_multiclass_segmentation_tf (2).ipynb](<Leather_unet_multiclass_segmentation_tf (2).ipynb>) | Alternate version of the segmentation notebook (low-light-robust UNet trained jointly on normal + gamma-degraded images, class-aware oversampling). |
| [notebooks/Leather_detection_segmentation.ipynb](notebooks/Leather_detection_segmentation.ipynb) | Notebook version of the classical CV segmentation pipeline (see `leather_defect_segmentation_cv.py`). |
| [leather_defect_segmentation_cv.py](leather_defect_segmentation_cv.py) | Pure image-processing baseline — **no ML model**. Reference-based anomaly detection, multi-scale Gaussian difference maps, Gabor filter bank, Laplacian-of-Gaussian, LAB color analysis, Otsu/percentile thresholding, morphological cleanup, per-defect-type tuned parameters. |
| [finetune_unet.py](finetune_unet.py) | Standalone (non-Colab) script to fine-tune the Attention U-Net locally — aggressive augmentation, focal+dice loss, class weighting, cosine-annealing LR with warm restarts. |
| [notebooks/Calculator.ipynb](notebooks/Calculator.ipynb) | Loads the trained model, predicts defect masks on new images, and computes the total damaged area per class. |
| [main.py](main.py) | Entry point for the live-detection GUI (CLI args, model path resolution). |
| [src/](src/) | GUI application package: `detector.py` (Attention U-Net wrapper + preprocessing), `worker.py` (QThread for capture + inference), `gui.py` (PyQt5 main window). |
| [requirements.txt](requirements.txt) | Python dependencies for the live-detection GUI. |
| `model/v1.keras` | Best checkpoint of the trained Attention U-Net segmentation model (previously `unet_results/best_attention_unet.keras`). |
| `unet_results_v3/clahe_comparison.png` | Visual comparison of images before/after CLAHE enhancement. |

## Dataset

Built on the **MVTec AD Leather** category:
- ~245 defect-free training images
- ~194 defective test images across 5 defect types: `color`, `cut`, `fold`, `glue`, `poke`
- Pixel-level ground-truth masks for defective images

Segmentation classes: `background(0)`, `color(1)`, `cut(2)`, `fold(3)`, `glue(4)`, `poke(5)`.

The raw dataset (`leather/`) and generated result folders (`segmentation_results/`, `segmentation_results_v2/`, `lowlight/`) are git-ignored — download the MVTec AD Leather category separately and place it as `leather/` at the repo root (`leather/train/good`, `leather/test/<defect_type>`, `leather/ground_truth/<defect_type>`).

Processed data produced by `notebooks/data_pipeline.ipynb` is stored as float32 `.npy` arrays under `processed_dataset/{train,val,test}/{images,masks,lowlight}/` (images in `[0,1]`, masks as binary `{0,1}`).

## Methodology

- **Low-light simulation:** gamma darkening + Gaussian noise + vignetting + uneven illumination, applied to clean images to create paired normal/low-light training data.
- **CLAHE:** contrast-limited adaptive histogram equalization on the L channel (LAB space) to recover contrast lost to low-light degradation before feeding images to the segmentation model.
- **DDPM synthetic generation:** a U-Net-based noise predictor trained to reverse a forward diffusion process, used to synthesize additional low-light defective samples and balance the otherwise scarce defect classes.
- **Segmentation model:** Attention U-Net with a pretrained ResNet50 encoder, Squeeze-Excite blocks and deep supervision; loss = 0.6 × weighted Focal Cross-Entropy + 0.4 × weighted Dice, with square-root inverse-frequency class weights to counter background dominance.
- **Classical CV baseline:** no training required — anomaly maps are built from differences against a reference "good" image (multi-scale Gaussian, Gabor texture energy, LoG edges, LAB color deviation), fused per defect type, then thresholded and cleaned up morphologically.
- **Evaluation metrics:** IoU and Dice (segmentation quality, overall and per-class), PSNR and SSIM (synthetic/low-light image quality), confusion matrix, and failure-case analysis on worst-performing samples.

## Getting started

The notebooks are designed to run on **Google Colab** (GPU, e.g. T4) and expect the dataset to be available via Google Drive. Pipeline order:

1. `notebooks/data_pipeline.ipynb`
2. `notebooks/Synthetic_image_generation.ipynb`
3. `Leather_unet_multiclass_segmentation_tf.ipynb`
4. `notebooks/Calculator.ipynb`

To run the classical CV baseline or fine-tune the model locally, use the standalone scripts:

```bash
python leather_defect_segmentation_cv.py
python finetune_unet.py
```

Update the `BASE_DIR` / `RESULTS_DIR` path constants at the top of each script to point to your local `leather/` dataset location before running.

### Dependencies

```
tensorflow
opencv-python
numpy
scikit-image
scikit-learn
matplotlib
pandas
```

The live-detection GUI has its own dependency list in [requirements.txt](requirements.txt) (see below).

## Live detection GUI

PyQt5 desktop application for live leather-defect segmentation using the
trained Attention U-Net. Accepts webcam, image, or video input and
overlays per-pixel defect predictions in real time, with per-class area
statistics and a PASS/FAIL verdict.

### Defect classes

| Class      | Color    |
|------------|----------|
| background | black    |
| color      | red      |
| cut        | green    |
| fold       | blue     |
| glue       | yellow   |
| poke       | magenta  |

### Installation

Requires Python 3.9 – 3.12.

```bash
# Create a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **TensorFlow / Keras version note.** The model was saved with Keras 3.x,
> so you need `tensorflow >= 2.16` (Keras 3 is bundled). If the model
> fails to load with layer-config errors, install the same TF version
> you used to train it.

### Providing the model

The app looks for a `.keras` model in this order — first match wins:

1. **`--model` CLI argument**
   ```bash
   python main.py --model model/v1.keras
   ```
2. **`LEATHER_MODEL_PATH` environment variable**
   ```bash
   # Linux / macOS
   LEATHER_MODEL_PATH=/path/to/model.keras python main.py
   # Windows (PowerShell)
   $env:LEATHER_MODEL_PATH = "C:\path\to\model.keras"; python main.py
   ```
3. **Default location: `models/v1.keras`**
   Drop your file there and just run `python main.py`.
4. **GUI button.** If none of the above resolve, the app still launches
   normally — click **Load model (.keras)…** in the top-left panel.

The trained model committed in this repository is at `model/v1.keras`.

### Running

```bash
python main.py
```

Workflow inside the app:

1. Wait for the model to load (status bar shows progress; takes 10–30 s
   the first time as TensorFlow compiles the graph).
2. Pick a **Source**: Webcam / Image / Video.
3. For Webcam: set the camera index (0 is usually the built-in camera).
   For Image / Video: click **Browse…** and pick the file.
4. Click **▶ Start**.
5. Adjust display:
   * **Display mode** — Overlay, Mask only, Side-by-side, Original
   * **Overlay opacity** — slider
   * **Defect classes** — checkboxes hide individual classes from the
     overlay (stats are unaffected)
6. Set **Pass/Fail threshold** (% defect area) — the right-hand badge
   turns red above the threshold, green below.
7. Optional: enter **Sample W/H** in mm to get a real-world mm²
   reading in the stats panel.
8. Click **📷 Save snapshot** to dump the current frame plus mask and
   stats.

### Snapshot outputs

Each snapshot produces five files in the chosen folder, all sharing the
same timestamp prefix `snapshot_YYYYMMDD_HHMMSS_`:

| Suffix             | Contents                                        |
|--------------------|-------------------------------------------------|
| `_original.png`    | Original BGR frame                              |
| `_overlay.png`     | Overlay with defects highlighted                |
| `_mask_color.png`  | Colorized mask (per-class colors)               |
| `_mask_raw.png`    | Raw integer class mask (0–5) for downstream use |
| `_stats.json`      | Per-class pixels/percent, verdict, threshold    |

### Performance

Per-frame inference time depends on hardware:

* **GPU (any CUDA-capable):** real-time at webcam frame rate (30+ FPS).
* **CPU only:** roughly 3–10 FPS at typical webcam resolution.

If CPU performance is unacceptable, ensure `tensorflow` was installed
with GPU support (`pip install tensorflow[and-cuda]` on Linux) and that
a GPU is visible (`tf.config.list_physical_devices('GPU')` is non-empty).

### A note on import order (Windows)

`main.py` and `src/gui.py` both import `src.detector` (which loads
TensorFlow) **before** PyQt5 and OpenCV. This is deliberate: on Windows,
PyQt5 and cv2 ship bundled MSVC runtime DLLs that, if loaded first,
prevent TF's native runtime from initialising (`DllMain returned FALSE`,
Windows error 0x45A). The fix is to load TF first so it resolves against
the system MSVC runtime; PyQt5/cv2 then reuse the same runtime.

Do not reorder the imports near the top of `main.py` or `src/gui.py`
without testing on Windows.

### Troubleshooting

* **Model fails to load with a `quantization_config` error.**
  `detector.py` already patches `keras.layers.Dense.__init__` to absorb
  this kwarg. If you still see the error, your TF/Keras version is too
  old — upgrade to `tensorflow >= 2.16`.
* **`DllMain returned FALSE` / TF fails to initialise on Windows.**
  See the import-order note above. Make sure `src.detector` is imported
  before PyQt5/cv2.
* **Webcam opens but no frames arrive.**
  Try a different camera index (0, 1, 2…). On Windows, some webcam
  drivers only expose certain backends; OpenCV will pick automatically
  but slowly. If it hangs, kill the app and try again.
* **First frame is very slow.**
  Expected — TF compiles the model graph on the first prediction.
  The detector runs one dummy prediction during `warmup()` to absorb
  this, but a small additional stall on the first real frame can still
  happen.
* **Video file plays slowly.**
  Video mode processes frames as fast as inference allows; it does not
  respect the video's native FPS. That is intentional (each frame is
  segmented, not just shown).
* **Snapshot button is disabled.**
  No frame has been processed yet. Wait for at least one prediction.
