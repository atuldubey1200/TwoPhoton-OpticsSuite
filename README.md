# TwoPhoton-OpticsSuite
A collection of Python-based tools for two-photon in vivo fluorescence imaging and optical system design in biophotonics research. Includes interactive GUIs for ray tracing, galvo scanner alignment, lens configuration optimization, and fluorescence signal analysis. Designed for use in advanced retinal imaging and optical alignment studies.
# Ray Trace GUI — Galvo + L1 + L2 + L2'

An **interactive optical ray tracing tool** built in Python using **PyQt5** and **Matplotlib**, designed for modeling **galvo-based scanning systems** in **two-photon in vivo fluorescence imaging** setups.

---

## Features
- Adjustable **galvo scan angle** and **semi-aperture**.
- Configurable **focal lengths** and **positions** for L1, L2, and L2'.
- **Afocal telescope mode** with one-click configuration.
- **Central-ray image plane** visualization.
- Export **ray path data to CSV** for analysis.
- Save **publication-quality plots** (Times New Roman font).

---

## Installation
```bash
pip install PyQt5 matplotlib numpy
# Intensity vs AOS GUI

A PyQt5-based application to compute and visualize **Intensity vs Area of Scanning (AOS)** from grayscale images.  
The tool is useful in imaging workflows (e.g., two-photon microscopy, SLO) where pixel-level intensity distributions need to be quantified.

---

## Features
- Load common image formats: PNG, JPG, BMP, TIFF
- Choose AOS calculation mode:
  - `x*y` (product of pixel indices)
  - `cumulative` (`(x+1)*(y+1)`)
- Option to restrict analysis to the left half of the image
- Interactive plot of **Mean Intensity vs AOS**
- Export results to CSV (`AOS_pixels, MeanIntensity`)

---

## Installation
Make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install numpy pillow matplotlib pyqt5
