# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 00:15:28 2025

@author: atuld
"""
import sys
import os
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QComboBox, QCheckBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

APP_FONT_FAMILY = "Times New Roman"
APP_FONT_SIZE = 10

def compute_aos_intensity(arr, use_left_half=True, aos_mode="x*y"):
    """Compute Intensity vs AOS for a grayscale numpy array.
    aos_mode: "x*y" or "cumulative"
    """
    if use_left_half:
        h, w = arr.shape
        arr = arr[:, : w // 2 if w > 1 else w]

    # indices
    y_idx, x_idx = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij")

    if aos_mode == "x*y":
        aos_vals = (x_idx * y_idx).astype(np.int64)
    else:
        # cumulative area from origin inclusive
        aos_vals = ((x_idx + 1) * (y_idx + 1)).astype(np.int64)

    intensities = arr.astype(np.float64).flatten()
    aos_flat = aos_vals.flatten()

    # group by AOS and compute mean intensity
    # to be efficient, sort by aos then group
    order = np.argsort(aos_flat)
    aos_sorted = aos_flat[order]
    intensity_sorted = intensities[order]

    # find group boundaries
    unique_aos, idx_starts = np.unique(aos_sorted, return_index=True)
    # compute means per group
    means = []
    n = len(aos_sorted)
    for i, start in enumerate(idx_starts):
        end = idx_starts[i + 1] if i + 1 < len(idx_starts) else n
        means.append(float(np.mean(intensity_sorted[start:end])))

    return unique_aos, np.array(means, dtype=float), arr

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        super().__init__(self.fig)

class AOSApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        QtWidgets.QApplication.setFont(QtGui.QFont(APP_FONT_FAMILY, APP_FONT_SIZE))
        self.setWindowTitle("Intensity vs AOS (pixels)")
        self.setMinimumSize(1000, 600)

        # widgets
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #111; color: #ddd; border: 1px solid #333;")
        self.image_label.setMinimumHeight(260)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)

        self.chk_left = QCheckBox("Use left half only")
        self.chk_left.setChecked(True)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["x*y", "cumulative"])
        self.combo_mode.setToolTip("Choose AOS definition: x*y or cumulative (x+1)*(y+1)")

        self.btn_compute = QPushButton("Compute & Plot")
        self.btn_compute.clicked.connect(self.compute_plot)

        self.btn_save = QPushButton("Save CSV")
        self.btn_save.clicked.connect(self.save_csv)
        self.btn_save.setEnabled(False)

        self.status = QLabel("Ready")
        self.status.setStyleSheet("color: #4caf50;")

        # matplotlib canvas
        self.canvas = MplCanvas(self, width=5.5, height=3.8, dpi=110)

        # layout
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.btn_load)
        top_controls.addWidget(self.chk_left)
        top_controls.addWidget(QLabel("AOS mode:"))
        top_controls.addWidget(self.combo_mode)
        top_controls.addStretch()
        top_controls.addWidget(self.btn_compute)
        top_controls.addWidget(self.btn_save)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top_controls)

        mid = QHBoxLayout()
        mid.addWidget(self.image_label, 1)
        mid.addWidget(self.canvas, 1)
        lay.addLayout(mid)
        lay.addWidget(self.status)

        self.setCentralWidget(central)

        # data holders
        self.image_path = None
        self.gray = None
        self.result_aos = None
        self.result_intensity = None

    def load_image(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not fpath:
            return
        try:
            im = Image.open(fpath).convert("L")
            arr = np.array(im)
            self.gray = arr
            self.image_path = fpath
            # show preview scaled
            qimg = QtGui.QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QtGui.QImage.Format_Grayscale8)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
            self.status.setText(f"Loaded: {os.path.basename(fpath)}  |  Shape: {arr.shape[1]} x {arr.shape[0]}")
            self.btn_save.setEnabled(False)
            self.canvas.ax.clear()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def compute_plot(self):
        if self.gray is None:
            QMessageBox.warning(self, "No image", "Please load an image first.")
            return
        try:
            use_left = self.chk_left.isChecked()
            mode = self.combo_mode.currentText()
            aos, mean_intensity, used_arr = compute_aos_intensity(self.gray, use_left_half=use_left, aos_mode=mode)

            # Store results
            self.result_aos = aos
            self.result_intensity = mean_intensity

            # Plot
            self.canvas.ax.clear()
            self.canvas.ax.plot(aos, mean_intensity)
            self.canvas.ax.set_xlabel("AOS (pixels)")
            self.canvas.ax.set_ylabel("Intensity (a.u.)")
            self.canvas.ax.set_title("Intensity vs AOS (pixels)")
            self.canvas.ax.grid(True, alpha=0.4)
            self.canvas.draw()

            self.status.setText(f"Computed with mode={mode}. Points={len(aos)}")
            self.btn_save.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Computation failed:\n{e}")

    def save_csv(self):
        if self.result_aos is None:
            QMessageBox.warning(self, "No data", "Compute first, then save.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "Save CSV", "intensity_vs_aos.csv", "CSV Files (*.csv)")
        if not fpath:
            return
        try:
            import csv
            with open(fpath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["AOS_pixels", "MeanIntensity"])
                for a, m in zip(self.result_aos, self.result_intensity):
                    writer.writerow([int(a), float(m)])
            self.status.setText(f"Saved CSV: {os.path.basename(fpath)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Intensity vs AOS")
    app.setFont(QtGui.QFont(APP_FONT_FAMILY, APP_FONT_SIZE))
    w = AOSApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
