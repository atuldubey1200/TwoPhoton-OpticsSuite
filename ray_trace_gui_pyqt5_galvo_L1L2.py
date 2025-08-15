# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 13:36:02 2025

@author: atuld
"""
"""
Ray Trace GUI — Galvo + L1 + L2 + L2'  (PyQt5 + Matplotlib)

What it models
--------------
- A galvo mirror pivot at z = 0 (origin). You set a *scan angle* θ (deg).
- L1 at position z_L1 with focal length f0. Rays originate at the galvo pivot and
  are constructed to reach evenly spaced heights on L1 in [-A, +A] (semi-aperture A).
  Then the bundle is rotated by θ, which shifts the hit positions on L1 as distance * tan(θ).
- L2 at z_L2 with focal f1, and L2' at z_L2p with focal f2 (a telescope section).
- All positions are absolute along the z-axis (mm). You can change all of them.
- Shows optional *central-ray image plane* (where the chief ray crosses y=0 after L2').

Notes
-----
- If z_L1 equals the galvo position (0), geometry is degenerate. The UI prevents this.
- If z_L1 ~= f0 (galvo at front focal plane), L1 outputs (approximately) collimated
  rays for θ = 0; with θ ≠ 0, the output remains collimated but tilted.
- “Afocal L2-L2'” button sets z_L2p = z_L2 + f1 + f2.

Run
---
    pip install PyQt5 matplotlib numpy
    python ray_trace_gui_pyqt5_galvo_L1L2.py

Or in Spyder:
    %runfile path/to/ray_trace_gui_pyqt5_galvo_L1L2.py
"""

import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QPushButton, QCheckBox, QFileDialog, QGroupBox, QFormLayout
)

import matplotlib
matplotlib.use("Qt5Agg")
# Use Times New Roman if available; will gracefully fallback if not
matplotlib.rcParams["font.family"] = "Times New Roman"
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


# ---------- Paraxial helpers ----------

def lens_refraction(theta, y, f):
    """Thin lens: theta' = theta - y/f ; y unchanged at lens plane."""
    return theta - y / f, y

def ray_segment(theta, y, z0, z1, npts=64):
    """Propagate a ray from z0 to z1 with slope theta (rad)."""
    z = np.linspace(z0, z1, npts)
    yline = y + theta * (z - z0)
    return z, yline

def propagate_to(theta, y, z0, z1):
    """Just compute (theta, y) at z1 from (theta, y) at z0."""
    return theta, y + theta * (z1 - z0)


# ---------- Main widget ----------

class RayTraceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ray Trace — Galvo + L1 + L2 + L2' (PyQt5)")
        self._build_ui()
        self._connect()
        self.update_plot()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(11, 3.2), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Controls
        ctrl = QGroupBox("Parameters")
        form = QFormLayout(ctrl)

        # Galvo / bundle at L1
        self.spin_theta = QDoubleSpinBox(); self.spin_theta.setRange(-30.0, 30.0); self.spin_theta.setDecimals(3)
        self.spin_theta.setValue(0.0); self.spin_theta.setSuffix(" deg")

        self.spin_A_L1 = QDoubleSpinBox(); self.spin_A_L1.setRange(0.1, 200.0); self.spin_A_L1.setDecimals(2)
        self.spin_A_L1.setValue(8.0); self.spin_A_L1.setSuffix(" mm")

        self.spin_nrays = QSpinBox(); self.spin_nrays.setRange(1, 4001); self.spin_nrays.setValue(11)

        # L1
        self.spin_f0 = QDoubleSpinBox(); self.spin_f0.setRange(1.0, 10000.0); self.spin_f0.setDecimals(2)
        self.spin_f0.setValue(75.0); self.spin_f0.setSuffix(" mm")

        self.spin_z_L1 = QDoubleSpinBox(); self.spin_z_L1.setRange(0.1, 100000.0); self.spin_z_L1.setDecimals(2)
        self.spin_z_L1.setValue(75.0); self.spin_z_L1.setSuffix(" mm")

        # L2
        self.spin_f1 = QDoubleSpinBox(); self.spin_f1.setRange(1.0, 10000.0); self.spin_f1.setDecimals(2)
        self.spin_f1.setValue(75.0); self.spin_f1.setSuffix(" mm")

        self.spin_z_L2 = QDoubleSpinBox(); self.spin_z_L2.setRange(0.1, 100000.0); self.spin_z_L2.setDecimals(2)
        self.spin_z_L2.setValue(200.0); self.spin_z_L2.setSuffix(" mm")

        # L2'
        self.spin_f2 = QDoubleSpinBox(); self.spin_f2.setRange(1.0, 10000.0); self.spin_f2.setDecimals(2)
        self.spin_f2.setValue(50.0); self.spin_f2.setSuffix(" mm")

        self.spin_z_L2p = QDoubleSpinBox(); self.spin_z_L2p.setRange(0.1, 100000.0); self.spin_z_L2p.setDecimals(2)
        self.spin_z_L2p.setValue(210.0); self.spin_z_L2p.setSuffix(" mm")

        # Toggles
        self.chk_afocal_telescope = QCheckBox("Set L2-L2' afocal (z_L2' = z_L2 + f1 + f2)")
        self.chk_show_focus = QCheckBox("Show central-ray image plane"); self.chk_show_focus.setChecked(True)

        # Drawing extents
        self.spin_z_left = QDoubleSpinBox(); self.spin_z_left.setRange(-100000.0, 0.0); self.spin_z_left.setDecimals(2)
        self.spin_z_left.setValue(-30.0); self.spin_z_left.setSuffix(" mm")

        self.spin_z_right = QDoubleSpinBox(); self.spin_z_right.setRange(10.0, 200000.0); self.spin_z_right.setDecimals(2)
        self.spin_z_right.setValue(400.0); self.spin_z_right.setSuffix(" mm")

        # Layout form
        form.addRow("Galvo scan angle θ:", self.spin_theta)
        form.addRow("Semi-aperture at L1:", self.spin_A_L1)
        form.addRow("Number of rays:", self.spin_nrays)

        form.addRow("L1 focal f0:", self.spin_f0)
        form.addRow("L1 position z_L1:", self.spin_z_L1)

        form.addRow("L2 focal f1:", self.spin_f1)
        form.addRow("L2 position z_L2:", self.spin_z_L2)

        form.addRow("L2' focal f2:", self.spin_f2)
        form.addRow("L2' position z_L2':", self.spin_z_L2p)

        form.addRow("", self.chk_afocal_telescope)
        form.addRow("", self.chk_show_focus)

        form.addRow("Draw from z =", self.spin_z_left)
        form.addRow("Draw to z =", self.spin_z_right)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_update = QPushButton("Update Plot")
        self.btn_save = QPushButton("Save Figure…")
        self.btn_export = QPushButton("Export Rays CSV…")
        btn_row.addWidget(self.btn_update); btn_row.addStretch(1); btn_row.addWidget(self.btn_save); btn_row.addWidget(self.btn_export)

        # Status
        self.lbl_report = QLabel("—")

        # Pack
        main.addWidget(self.toolbar)
        main.addWidget(self.canvas)
        main.addWidget(ctrl)
        main.addLayout(btn_row)
        main.addWidget(self.lbl_report)

    def _connect(self):
        self.btn_update.clicked.connect(self.update_plot)
        self.btn_save.clicked.connect(self.save_figure)
        self.btn_export.clicked.connect(self.export_csv)
        self.chk_afocal_telescope.stateChanged.connect(self._do_afocal)

        # live updates
        for w in [self.spin_theta, self.spin_A_L1, self.spin_nrays,
                  self.spin_f0, self.spin_z_L1, self.spin_f1, self.spin_z_L2,
                  self.spin_f2, self.spin_z_L2p, self.spin_z_left, self.spin_z_right,
                  self.chk_show_focus]:
            try:
                w.valueChanged.connect(self.update_plot)
            except Exception:
                w.stateChanged.connect(self.update_plot)

    # --- core computation ---

    def build_initial_rays(self, z_galvo, z_L1, A_L1, n_rays, theta_scan_rad):
        """
        Build rays that *originate at the galvo pivot* (z=z_galvo, y=0) and would reach
        evenly spaced heights on L1 in [-A_L1, +A_L1] when theta_scan_rad = 0.
        Then apply a rotation (add slope) equal to tan(theta_scan_rad) to the whole bundle,
        which shifts their heights on L1 by (z_L1 - z_galvo) * tan(theta).
        """
        dz = z_L1 - z_galvo
        if dz <= 0:
            dz = 1e-6  # avoid division by zero; UI guards but just in case

        if n_rays == 1:
            y_targets = np.array([0.0])
        else:
            y_targets = np.linspace(-A_L1, +A_L1, n_rays)

        slope_offset = np.tan(theta_scan_rad)
        thetas0 = y_targets / dz + slope_offset  # slopes at the galvo (rad)
        y0 = np.zeros_like(thetas0)              # all start at y=0 at the pivot
        return thetas0, y0

    def trace_system(self, thetas0, y0, z_left, z_L1, f0, z_L2, f1, z_L2p, f2, z_right):
        """
        Trace each ray: galvo -> L1 -> L2 -> L2' -> right edge.
        Return a list of polylines [(z, y), ...] and the central-ray focus (if any).
        """
        polylines = []

        # propagate each ray
        for th0, ystart in zip(thetas0, y0):
            # Segment G: z_left to pivot (draw the left line backwards with same slope)
            zG, yG = ray_segment(th0, ystart, 0.0, z_left, npts=32)  # leftwards
            # Segment to L1
            zA, yA = ray_segment(th0, yG[-1], z_left, z_L1, npts=96)
            th1, y1 = lens_refraction(th0, yA[-1], f0)

            # L1 -> L2
            zB, yB = ray_segment(th1, y1, z_L1, z_L2, npts=160)
            th2, y2 = lens_refraction(th1, yB[-1], f1)

            # L2 -> L2'
            zC, yC = ray_segment(th2, y2, z_L2, z_L2p, npts=160)
            th3, y3 = lens_refraction(th2, yC[-1], f2)

            # L2' -> right
            zD, yD = ray_segment(th3, y3, z_L2p, z_right, npts=200)

            # Merge polylines (ensure continuity without duplicate vertices)
            z = np.concatenate([zA, zB[1:], zC[1:], zD[1:]])
            y = np.concatenate([yA, yB[1:], yC[1:], yD[1:]])
            polylines.append((z, y))

        # central-ray image plane: take the "middle" ray (closest to y_target=0 at θ=0)
        mid = len(thetas0) // 2
        # recompute its final state exactly to solve for crossing y=0 after L2'
        th0 = thetas0[mid]; y_mid0 = y0[mid]
        # to L1
        _, yL1 = propagate_to(th0, y_mid0, 0.0, z_L1)
        th1_mid = th0 - yL1 / f0
        # to L2
        _, yL2 = propagate_to(th1_mid, yL1, z_L1, z_L2)
        th2_mid = th1_mid - yL2 / f1
        # to L2'
        _, yL2p = propagate_to(th2_mid, yL2, z_L2, z_L2p)
        th3_mid = th2_mid - yL2p / f2

        # find z where y=0 after L2'
        focus_z = None
        if abs(th3_mid) > 1e-12:
            focus_z = z_L2p - yL2p / th3_mid  # y(z) = yL2p + th3_mid*(z - z_L2p) = 0

        return polylines, focus_z

    def update_plot(self):
        # read UI
        theta_deg = self.spin_theta.value()
        A_L1 = self.spin_A_L1.value()
        n_rays = self.spin_nrays.value()

        f0 = self.spin_f0.value()
        z_L1 = self.spin_z_L1.value()

        f1 = self.spin_f1.value()
        z_L2 = self.spin_z_L2.value()

        f2 = self.spin_f2.value()
        z_L2p = self.spin_z_L2p.value()

        z_left = self.spin_z_left.value()
        z_right = self.spin_z_right.value()

        # build rays and trace
        thetas0, y0 = self.build_initial_rays(0.0, z_L1, A_L1, n_rays, np.deg2rad(theta_deg))
        polylines, focus_z = self.trace_system(thetas0, y0, z_left, z_L1, f0, z_L2, f1, z_L2p, f2, z_right)

        # plot
        self.ax.clear()

        # rays
        for (z, y) in polylines:
            self.ax.plot(z, y, linewidth=1.4)

        # elements
        self.ax.axvline(0.0, linestyle="--", linewidth=2)  # galvo
        self.ax.text(0.0, 0.92*self._auto_ylim(polylines), "Galvo", ha="center", va="bottom", fontsize=9)

        self.ax.axvline(z_L1, linestyle="--", linewidth=2)
        self.ax.text(z_L1, 0.92*self._auto_ylim(polylines), f"L1 (f0={f0:.1f} mm)", ha="center", va="bottom", fontsize=9)

        self.ax.axvline(z_L2, linestyle="--", linewidth=2)
        self.ax.text(z_L2, 0.92*self._auto_ylim(polylines), f"L2 (f1={f1:.1f} mm)", ha="center", va="bottom", fontsize=9)

        self.ax.axvline(z_L2p, linestyle="--", linewidth=2)
        self.ax.text(z_L2p, 0.92*self._auto_ylim(polylines), f"L2' (f2={f2:.1f} mm)", ha="center", va="bottom", fontsize=9)

        # focus marker
        report = []
        if self.chk_show_focus.isChecked():
            if focus_z is None or not (z_L2p <= focus_z <= z_right + 1e-9):
                report.append("Central-ray image plane: collimated or out of range")
            else:
                self.ax.axvline(focus_z, color="k", linestyle=":", linewidth=1.3)
                self.ax.plot([focus_z], [0], marker="o", ms=4, color="k")
                report.append(f"Central-ray image plane ≈ {focus_z:.3f} mm (absolute z)")

        # cosmetics
        self.ax.set_title("Galvo→L1→L2→L2' Ray Trace (paraxial)")
        self.ax.set_xlabel("Optical axis z (mm)")
        self.ax.set_ylabel("Height y (mm)")
        self.ax.set_xlim([z_left, z_right])
        # y-limits based on polylines content
        ymag = self._auto_ylim(polylines)
        self.ax.set_ylim([-ymag, ymag])
        self.ax.grid(True, linestyle=":", linewidth=0.7)
        self.canvas.draw_idle()

        # status text
        # report collimation condition at L1 (theta after L1 for central ray when θ=0)
        dz = z_L1 - 0.0
        slope_off = np.tan(np.deg2rad(theta_deg))
        # central ray uses target y=0 at θ=0 => base slope = 0; with scan => th0 = slope_off
        th0_central = slope_off
        _, yL1_c = propagate_to(th0_central, 0.0, 0.0, z_L1)
        th_after_L1 = th0_central - yL1_c / f0  # if dz==f0 and θ=0 -> 0
        report.append(f"θ after L1 (central ray) = {np.rad2deg(th_after_L1):.4f}°")

        self.lbl_report.setText(" | ".join(report))

    def _auto_ylim(self, polylines):
        # Compute a symmetric y-limit scaling from all segments
        ymin, ymax = 0.0, 0.0
        for _, y in polylines:
            ymin = min(ymin, np.min(y))
            ymax = max(ymax, np.max(y))
        # pad a bit
        pad = 0.08 * max(1.0, ymax - ymin)
        return max(abs(ymin), abs(ymax)) + pad

    def _do_afocal(self, _state):
        if self.chk_afocal_telescope.isChecked():
            zL2 = self.spin_z_L2.value()
            f1 = self.spin_f1.value()
            f2 = self.spin_f2.value()
            self.spin_z_L2p.blockSignals(True)
            self.spin_z_L2p.setValue(zL2 + f1 + f2)
            self.spin_z_L2p.blockSignals(False)
        self.update_plot()

    # --- utility actions ---

    def save_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "ray_trace.png", "PNG Image (*.png);;PDF (*.pdf)")
        if path:
            self.fig.savefig(path, bbox_inches="tight", dpi=300)

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Rays CSV", "rays.csv", "CSV (*.csv)")
        if not path:
            return
        # rebuild current curves to export
        theta_deg = self.spin_theta.value()
        A_L1 = self.spin_A_L1.value()
        n_rays = self.spin_nrays.value()
        f0 = self.spin_f0.value(); z_L1 = self.spin_z_L1.value()
        f1 = self.spin_f1.value(); z_L2 = self.spin_z_L2.value()
        f2 = self.spin_f2.value(); z_L2p = self.spin_z_L2p.value()
        z_left = self.spin_z_left.value(); z_right = self.spin_z_right.value()

        thetas0, y0 = self.build_initial_rays(0.0, z_L1, A_L1, n_rays, np.deg2rad(theta_deg))
        polylines, focus_z = self.trace_system(thetas0, y0, z_left, z_L1, f0, z_L2, f1, z_L2p, f2, z_right)

        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ray_id", "z_mm", "y_mm"])
            for ridx, (z, y) in enumerate(polylines):
                for zi, yi in zip(z, y):
                    writer.writerow([ridx, zi, yi])


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = RayTraceWidget()
    w.resize(1200, 740)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
