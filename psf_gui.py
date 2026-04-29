"""
psfScope GUI — Graphical interface for estimate_psf_from_beads.

Four tabs
---------
1. Estimation  — parameters, progress bar, log
2. PSF         — XY / XZ / YZ cross-sections with FWHM readout
3. Beads       — detected bead map and FWHM histograms
4. FOV Map     — spatial variation of resolution across the field of view

Usage
-----
    python psf_gui.py
    # or launch from postprocess_psf.py with no CLI arguments
"""

import csv
import os
import sys
import glob as _glob
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit

from postprocess_psf import estimate_psf_from_beads, measure_fwhm_from_averaged_psf

PAD = {"padx": 8, "pady": 4}


# =============================================================================
# Main GUI class
# =============================================================================

class PSFScopeGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("psfScope")
        self.root.minsize(960, 680)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._psf            = None
        self._bead_data      = None
        self._queue          = queue.Queue()
        self._fov_cbar       = None
        self._last_tif_path  = None

        # Theoretical PSF parameters (Born-Wolf model)
        self.lambda_var      = tk.StringVar(value="515")   # emission wavelength (nm)
        self.na_var          = tk.StringVar(value="1.1")   # numerical aperture
        self.n_var           = tk.StringVar(value="1.33")  # refractive index (water)
        self.show_theory_var = tk.BooleanVar(value=False)

        self._build_ui()

    # =========================================================================
    # Top-level layout
    # =========================================================================

    def _build_ui(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Tab 1: Estimation
        f1 = ttk.Frame(self.nb)
        self.nb.add(f1, text="  Estimation  ")
        self._build_estimation_tab(f1)

        # Tab 2: PSF cross-sections
        f2 = ttk.Frame(self.nb)
        self.nb.add(f2, text="  PSF  ")
        self._build_psf_tab(f2)

        # Tab 3: Detected beads
        f3 = ttk.Frame(self.nb)
        self.nb.add(f3, text="  Beads  ")
        self._build_beads_tab(f3)

        # Tab 4: FOV resolution map
        f4 = ttk.Frame(self.nb)
        self.nb.add(f4, text="  FOV Map  ")
        self._build_fov_tab(f4)

        # Tab 5: FWHM diagnostics
        f5 = ttk.Frame(self.nb)
        self.nb.add(f5, text="  FWHM diagnostics  ")
        self._build_hist_fit_tab(f5)

        # Global status bar
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief="sunken", anchor="w").grid(
            row=1, column=0, sticky="ew", padx=4, pady=2)

    # =========================================================================
    # Tab 1: Estimation
    # =========================================================================

    def _build_estimation_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        # --- Input mode ---
        mf = ttk.LabelFrame(parent, text="Input mode")
        mf.grid(row=0, column=0, sticky="ew", **PAD)
        self.mode_var = tk.StringVar(value="file")

        def _upd_browse():
            self.browse_btn.config(
                text="Select file" if self.mode_var.get() == "file" else "Select folder"
            )

        ttk.Radiobutton(mf, text="Single file", variable=self.mode_var,
                        value="file",   command=_upd_browse).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(mf, text="Folder",      variable=self.mode_var,
                        value="folder", command=_upd_browse).grid(row=0, column=1, padx=10)

        # --- Files ---
        io = ttk.LabelFrame(parent, text="Files")
        io.grid(row=1, column=0, sticky="ew", **PAD)
        io.columnconfigure(1, weight=1)

        self.input_var  = tk.StringVar()
        self.output_var = tk.StringVar()

        def _browse_in():
            if self.mode_var.get() == "file":
                p = filedialog.askopenfilename(
                    title="Select bead TIFF",
                    filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
                )
            else:
                p = filedialog.askdirectory(title="Select folder containing TIFFs")
            if p:
                self.input_var.set(p)

        def _browse_out():
            p = filedialog.asksaveasfilename(
                title="Save PSF as …",
                defaultextension=".tif",
                filetypes=[("TIFF files", "*.tif *.tiff")],
            )
            if p:
                self.output_var.set(p)

        ttk.Label(io, text="Input:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Entry(io, textvariable=self.input_var, width=60).grid(row=0, column=1, sticky="ew", **PAD)
        self.browse_btn = ttk.Button(io, text="Select file", command=_browse_in)
        self.browse_btn.grid(row=0, column=2, **PAD)

        ttk.Label(io, text="Output:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Entry(io, textvariable=self.output_var, width=60).grid(row=1, column=1, sticky="ew", **PAD)
        ttk.Button(io, text="Save as …", command=_browse_out).grid(row=1, column=2, **PAD)
        ttk.Label(io, text="(empty → same directory as input, with _psf.tif suffix)",
                  foreground="gray").grid(row=2, column=1, sticky="w", padx=8)

        # --- Parameters + Theoretical PSF (side-by-side) ---
        param_row = ttk.Frame(parent)
        param_row.grid(row=2, column=0, sticky="ew")
        param_row.columnconfigure(0, weight=1)
        param_row.columnconfigure(1, weight=1)

        pf = ttk.LabelFrame(param_row, text="Parameters")
        pf.grid(row=0, column=0, sticky="nsew", **PAD)

        self.dx_var            = tk.StringVar(value="0.127")
        self.dz_var            = tk.StringVar(value="0.110")
        self.thr_var           = tk.StringVar(value="")
        self.sep_var           = tk.StringVar(value="2.0")
        self.margin_px_var     = tk.StringVar(value="2")
        self.r2_thresh_var     = tk.StringVar(value="0.9")
        self.rm_avg_var    = tk.BooleanVar(value=True)
        self.rm_mean_var   = tk.BooleanVar(value=False)
        self.rm_median_var = tk.BooleanVar(value=False)
        self.roi_z_var         = tk.StringVar(value="2.5")
        self.roi_y_var         = tk.StringVar(value="2.5")
        self.roi_x_var         = tk.StringVar(value="2.5")

        for lbl, var, r, c in [
            ("dx (µm):",                  self.dx_var,          0, 0),
            ("dz (µm):",                  self.dz_var,          0, 2),
            ("Threshold (auto if empty):", self.thr_var,        1, 0),
            ("Min separation (µm):",       self.sep_var,        1, 2),
            ("Edge margin (px):",          self.margin_px_var,  2, 0),
            ("R² threshold:",              self.r2_thresh_var,  2, 2),
        ]:
            ttk.Label(pf, text=lbl).grid(row=r, column=c,   sticky="w", **PAD)
            ttk.Entry(pf, textvariable=var, width=10).grid(row=r, column=c+1, sticky="w", **PAD)

        # Reporting mode: multi-select checkboxes
        ttk.Label(pf, text="Report:").grid(row=3, column=0, sticky="w", **PAD)
        ttk.Checkbutton(pf, text="avg-PSF",
                        variable=self.rm_avg_var).grid(
            row=3, column=1, sticky="w", padx=4)
        ttk.Checkbutton(pf, text="per-bead mean±SD",
                        variable=self.rm_mean_var).grid(
            row=3, column=2, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(pf, text="per-bead median±MAD",
                        variable=self.rm_median_var).grid(
            row=3, column=4, columnspan=2, sticky="w", padx=4)

        for lbl, var, c in [
            ("ROI Z (µm):", self.roi_z_var, 0),
            ("ROI Y (µm):", self.roi_y_var, 2),
            ("ROI X (µm):", self.roi_x_var, 4),
        ]:
            ttk.Label(pf, text=lbl).grid(row=4, column=c,   sticky="w", **PAD)
            ttk.Entry(pf, textvariable=var, width=10).grid(row=4, column=c+1, sticky="w", **PAD)

        ttk.Label(pf,
                  text="ROI: extraction window around each bead. "
                       "Reduce ROI Z (e.g. 1.2) for thin volumes.",
                  foreground="gray").grid(row=5, column=0, columnspan=6, sticky="w", padx=8)

        # Fitting mode
        fit_lbl = ttk.Label(pf, text="Fitting mode:")
        fit_lbl.grid(row=6, column=0, sticky="w", **PAD)

        self.fit_mode_var = tk.StringVar(value="1d")
        ttk.Radiobutton(pf, text="1D sequential  (fast)",
                        variable=self.fit_mode_var, value="1d").grid(
            row=6, column=1, columnspan=2, sticky="w", padx=4)
        ttk.Radiobutton(pf, text="3D simultaneous  (accurate, slower)",
                        variable=self.fit_mode_var, value="3d").grid(
            row=6, column=3, columnspan=3, sticky="w", padx=4)

        ttk.Label(pf,
                  text="3D fits a full Gaussian to the ROI volume — more accurate for "
                       "asymmetric PSFs but ~10–100× slower per bead.",
                  foreground="gray").grid(row=7, column=0, columnspan=6, sticky="w", padx=8)

        # --- Theoretical PSF (optional) ---
        tf = ttk.LabelFrame(param_row, text="Theoretical PSF (optional)")
        tf.grid(row=0, column=1, sticky="nsew", **PAD)

        for lbl, var, r in [
            ("λ_em (nm):", self.lambda_var, 0),
            ("NA:",        self.na_var,     1),
            ("n:",         self.n_var,      2),
        ]:
            ttk.Label(tf, text=lbl).grid(row=r, column=0, sticky="w", **PAD)
            ttk.Entry(tf, textvariable=var, width=9).grid(row=r, column=1, sticky="w", **PAD)

        ttk.Label(tf,
                  text="Born-Wolf model:\n"
                       "FWHM_xy = 0.51 · λ / NA\n"
                       "FWHM_z  = 0.887 · λ / (n − √(n²−NA²))",
                  foreground="gray", justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        ttk.Checkbutton(tf, text="Show theoretical overlay",
                        variable=self.show_theory_var,
                        command=self._refresh_theory_overlay).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # --- Run button + progress bar ---
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=3, column=0, sticky="ew", **PAD)

        self.run_btn = ttk.Button(ctrl, text="▶  Run PSF estimation",
                                  command=self._run)
        self.run_btn.pack(side="left", padx=8)

        self.clear_btn = ttk.Button(ctrl, text="✕  Clear",
                                    command=self._clear_results, state="disabled")
        self.clear_btn.pack(side="left", padx=4)

        self.progress = ttk.Progressbar(ctrl, length=380, mode="determinate",
                                        maximum=100)
        self.progress.pack(side="left", padx=8)

        self.pct_lbl = ttk.Label(ctrl, text="", width=5)
        self.pct_lbl.pack(side="left")

        self.save_res_btn = ttk.Button(ctrl, text="Save results …",
                                       command=self._save_results, state="disabled")
        self.save_res_btn.pack(side="right", padx=4)

        self.load_res_btn = ttk.Button(ctrl, text="Load results …",
                                       command=self._load_results)
        self.load_res_btn.pack(side="right", padx=4)

        # --- Log ---
        lf = ttk.LabelFrame(parent, text="Log")
        lf.grid(row=4, column=0, sticky="nsew", **PAD)
        parent.rowconfigure(4, weight=1)

        log_toolbar = ttk.Frame(lf)
        log_toolbar.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Button(log_toolbar, text="Save log …",
                   command=self._save_log).pack(side="right")

        self.log = ScrolledText(lf, height=12, state="disabled",
                                font=("Consolas", 9), wrap="word")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    # =========================================================================
    # Tab 2: PSF cross-sections
    # =========================================================================

    def _build_psf_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.psf_fig = Figure(figsize=(11, 4.5), tight_layout=True)
        self.ax_xy   = self.psf_fig.add_subplot(131)
        self.ax_xz   = self.psf_fig.add_subplot(132)
        self.ax_yz   = self.psf_fig.add_subplot(133)
        for ax, t in zip([self.ax_xy, self.ax_xz, self.ax_yz],
                         ["XY  (focal plane)",
                          "XZ  (axial · lateral X)",
                          "YZ  (axial · lateral Y)"]):
            ax.set_title(t, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=10)

        self.psf_canvas = FigureCanvasTkAgg(self.psf_fig, master=parent)
        self.psf_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb_frame = ttk.Frame(parent)
        tb_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.psf_canvas, tb_frame)

        self.psf_fwhm_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.psf_fwhm_var,
                  font=("Consolas", 9), foreground="#1a4080").grid(
            row=2, column=0, sticky="w", padx=10, pady=4)

        self.psf_theory_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.psf_theory_var,
                  font=("Consolas", 9), foreground="#b84800").grid(
            row=3, column=0, sticky="w", padx=10, pady=2)

        ttk.Button(parent, text="⬇  Save projections as TIFF",
                   command=self._save_psf_projections).grid(
            row=4, column=0, sticky="w", padx=10, pady=6)

    # =========================================================================
    # Tab 3: Detected beads
    # =========================================================================

    def _build_beads_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)

        # Left panel: bead position scatter
        self.beads_fig = Figure(figsize=(7, 5.5), tight_layout=True)
        self.beads_ax  = self.beads_fig.add_subplot(111)
        self.beads_ax.set_title("Bead positions (XY projection)", fontsize=9)
        self.beads_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=self.beads_ax.transAxes, color="gray")

        self.beads_canvas = FigureCanvasTkAgg(self.beads_fig, master=parent)
        self.beads_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.beads_canvas.mpl_connect('button_press_event', self._on_beads_click)

        tb1 = ttk.Frame(parent)
        tb1.grid(row=1, column=0, sticky="ew")
        self.beads_toolbar = NavigationToolbar2Tk(self.beads_canvas, tb1)

        ttk.Label(parent, text="Click on a bead to inspect its ROI and Gaussian fit",
                  foreground="gray", font=("TkDefaultFont", 8)).grid(
            row=2, column=0, sticky="w", padx=8, pady=2)

        # Right panel: FWHM histograms
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.hist_fig  = Figure(figsize=(4.5, 7.5), tight_layout=True)
        self.hist_ax1  = self.hist_fig.add_subplot(311)
        self.hist_ax2  = self.hist_fig.add_subplot(312)
        self.hist_ax3  = self.hist_fig.add_subplot(313)
        self.hist_ax1.set_title("FWHM_xy", fontsize=9)
        self.hist_ax2.set_title("FWHM_z",  fontsize=9)
        self.hist_ax3.set_title("FWHM vs bead Z position", fontsize=9)

        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=right)
        self.hist_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.beads_stats_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.beads_stats_var,
                  font=("Consolas", 8)).grid(row=1, column=0, sticky="w", padx=6, pady=2)

        ttk.Button(right, text="⬇  Export bead table (CSV)",
                   command=self._export_csv).grid(row=2, column=0, pady=6)

    # =========================================================================
    # Tab 4: FOV resolution map
    # =========================================================================

    def _build_fov_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # Controls
        ctrl = ttk.Frame(parent)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        ttk.Label(ctrl, text="Metric:").pack(side="left")
        self.fov_metric = tk.StringVar(value="FWHM_xy")
        cb = ttk.Combobox(ctrl, textvariable=self.fov_metric, width=14,
                          state="readonly",
                          values=["FWHM_xy", "FWHM_z", "FWHM_x", "FWHM_y",
                                  "Ellipticity", "SNR"])
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_fov())

        self.fov_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl,
                        text="Show all accepted beads (not only those used in PSF)",
                        variable=self.fov_all_var,
                        command=self._refresh_fov).pack(side="left", padx=10)

        # Figure
        self.fov_fig = Figure(figsize=(8, 6), tight_layout=True)
        self.fov_ax  = self.fov_fig.add_subplot(111)
        self.fov_ax.set_title("PSF variation across the field of view", fontsize=10)
        self.fov_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.fov_ax.transAxes, color="gray")

        self.fov_canvas = FigureCanvasTkAgg(self.fov_fig, master=parent)
        self.fov_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb2 = ttk.Frame(parent)
        tb2.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.fov_canvas, tb2)

    # =========================================================================
    # Tab 5: FWHM histogram fit
    # =========================================================================

    def _build_hist_fit_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # 2 rows × 3 cols:  top = avg-PSF profiles, bottom = per-bead histograms
        self.hfit_fig = Figure(figsize=(13, 7), tight_layout=True)
        self.hfit_prof_z = self.hfit_fig.add_subplot(231)
        self.hfit_prof_y = self.hfit_fig.add_subplot(232)
        self.hfit_prof_x = self.hfit_fig.add_subplot(233)
        self.hfit_hist_z = self.hfit_fig.add_subplot(234)
        self.hfit_hist_y = self.hfit_fig.add_subplot(235)
        self.hfit_hist_x = self.hfit_fig.add_subplot(236)

        for ax, title in zip(
            [self.hfit_prof_z, self.hfit_prof_y, self.hfit_prof_x],
            ["Profile Z  (avg PSF)", "Profile Y  (avg PSF)", "Profile X  (avg PSF)"],
        ):
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        for ax, title in zip(
            [self.hfit_hist_z, self.hfit_hist_y, self.hfit_hist_x],
            ["FWHM_z  per bead", "FWHM_y  per bead", "FWHM_x  per bead"],
        ):
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        self.hfit_canvas = FigureCanvasTkAgg(self.hfit_fig, master=parent)
        self.hfit_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb = ttk.Frame(parent)
        tb.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.hfit_canvas, tb)

    # =========================================================================
    # Run logic
    # =========================================================================

    def _parse_params(self):
        dx             = float(self.dx_var.get())
        dz             = float(self.dz_var.get())
        thr            = float(self.thr_var.get()) if self.thr_var.get().strip() else None
        sep            = float(self.sep_var.get())
        margin_px      = int(self.margin_px_var.get())
        r2_threshold   = float(self.r2_thresh_var.get())
        reporting_mode = [m for m, v in [
            ('averaged_psf',    self.rm_avg_var.get()),
            ('per_bead_mean',   self.rm_mean_var.get()),
            ('per_bead_median', self.rm_median_var.get()),
        ] if v]
        if not reporting_mode:
            raise ValueError("Select at least one reporting mode.")
        roi      = (float(self.roi_z_var.get()),
                    float(self.roi_y_var.get()),
                    float(self.roi_x_var.get()))
        fit_mode = self.fit_mode_var.get()
        return dx, dz, thr, sep, margin_px, r2_threshold, reporting_mode, roi, fit_mode

    def _run(self):
        inp = self.input_var.get().strip()
        if not inp:
            messagebox.showerror("Error", "Please specify an input file or folder.")
            return

        try:
            dx, dz, thr, sep, margin_px, r2_threshold, reporting_mode, roi, fit_mode = self._parse_params()
        except ValueError as e:
            messagebox.showerror("Parameter error", str(e))
            return

        outp = self.output_var.get().strip() or None

        if self.mode_var.get() == "file":
            tif_files = [inp]
        else:
            tif_files = sorted(
                _glob.glob(os.path.join(inp, "*.tif")) +
                _glob.glob(os.path.join(inp, "*.tiff"))
            )
            if not tif_files:
                messagebox.showerror("Error", "No .tif / .tiff files found in the selected folder.")
                return

        self.run_btn.config(state="disabled")
        self.progress["value"] = 0
        self.pct_lbl.config(text="0%")
        self._clear_log()

        def _worker():
            old_stdout = sys.stdout
            sys.stdout = _StdoutRedirector(self._log_append)
            try:
                n            = len(tif_files)
                results      = []
                ok_tif_files = []
                for i, tif in enumerate(tif_files):
                    frac_base  = i / n
                    frac_scale = 1.0 / n

                    def _cb(f, msg, _b=frac_base, _s=frac_scale):
                        self._queue.put(("progress", _b + f * _s, msg))

                    save = outp if (outp and n == 1) else None
                    try:
                        psf, save_path, bead_data = estimate_psf_from_beads(
                            tif_path          = tif,
                            dx                = dx,
                            dz                = dz,
                            threshold         = thr,
                            min_sep_um        = sep,
                            roi_um            = roi,
                            margin_px         = margin_px,
                            r2_threshold      = r2_threshold,
                            reporting_mode    = reporting_mode,
                            save_path         = save,
                            fit_mode          = fit_mode,
                            verbose           = True,
                            progress_callback = _cb,
                            return_bead_data  = True,
                        )
                        results.append((psf, bead_data, save_path))
                        ok_tif_files.append(tif)
                    except Exception as exc:
                        import traceback as _tb
                        print(f"\n⚠ Skipping {os.path.basename(tif)}: {exc}\n"
                              f"{_tb.format_exc()}")

                if not results:
                    raise RuntimeError(
                        "All volumes failed — no valid beads found in any file."
                    )

                if n == 1:
                    psf, bead_data, save_path = results[0]
                    last_tif = tif_files[0]
                else:
                    psf       = PSFScopeGUI._merge_psfs(results)
                    bead_data = PSFScopeGUI._merge_bead_data(results, ok_tif_files)
                    n_ok      = len(ok_tif_files)
                    try:
                        PSFScopeGUI._apply_combined_fwhm(psf, bead_data, dz, dx)
                        n_used_total = bead_data['n_used']
                        print(f"\n[PSF] ── Combined FWHM ({n_ok} volumes, "
                              f"{n_used_total} beads) ──")
                        for _axis, _avg, _mn, _sd, _med, _mad in [
                            ('z', bead_data['fwhm_averaged_psf_z'],
                             bead_data['fwhm_per_bead_mean_z'], bead_data['fwhm_per_bead_sd_z'],
                             bead_data['fwhm_median_z'], bead_data['fwhm_mad_z']),
                            ('y', bead_data['fwhm_averaged_psf_y'],
                             bead_data['fwhm_per_bead_mean_y'], bead_data['fwhm_per_bead_sd_y'],
                             bead_data['fwhm_median_y'], bead_data['fwhm_mad_y']),
                            ('x', bead_data['fwhm_averaged_psf_x'],
                             bead_data['fwhm_per_bead_mean_x'], bead_data['fwhm_per_bead_sd_x'],
                             bead_data['fwhm_median_x'], bead_data['fwhm_mad_x']),
                        ]:
                            _avg_str = f"{_avg:.0f}" if np.isfinite(_avg) else "?"
                            print(f"[PSF] FWHM_{_axis} = {_avg_str} nm (avg-PSF)   "
                                  f"per-bead: mean={_mn:.0f}±{_sd:.0f} SD, "
                                  f"median={_med:.0f} ±{_mad:.0f} MAD, N={n_used_total}")
                    except Exception as _exc:
                        print(f"[PSF] ⚠ Warning: combined FWHM computation failed: {_exc}")
                    skipped   = n - n_ok
                    save_path = (
                        f"{n_ok} volumes merged"
                        + (f"  ({skipped} skipped)" if skipped else "")
                    )
                    last_tif  = None   # inspector requires a single source file

                self._queue.put(("done", psf, bead_data, save_path, last_tif))
            except Exception:
                import traceback
                self._queue.put(("error", traceback.format_exc()))
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self):
        try:
            while True:
                msg  = self._queue.get_nowait()
                kind = msg[0]

                if kind == "progress":
                    _, frac, text = msg
                    pct = int(frac * 100)
                    self.progress["value"] = pct
                    self.pct_lbl.config(text=f"{pct}%")
                    self.status_var.set(text)

                elif kind == "done":
                    _, psf, bead_data, save_path, tif_path = msg
                    self._psf           = psf
                    self._bead_data     = bead_data
                    self._last_tif_path = tif_path
                    self.progress["value"] = 100
                    self.pct_lbl.config(text="100%")
                    self.status_var.set(f"✓  {save_path}")
                    self.run_btn.config(state="normal")
                    self.clear_btn.config(state="normal")
                    self.save_res_btn.config(state="normal")
                    self._update_all_plots()
                    return

                elif kind == "error":
                    _, tb = msg
                    self._log_append(f"\n✗ ERROR:\n{tb}")
                    self.status_var.set("✗ Error — see log")
                    self.run_btn.config(state="normal")
                    return

        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # =========================================================================
    # Log / Results I/O
    # =========================================================================

    def _save_log(self):
        path = filedialog.asksaveasfilename(
            title="Save log as …",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        text = self.log.get("1.0", "end-1c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        self.status_var.set(f"Log saved → {os.path.basename(path)}")

    def _save_results(self):
        """Serialize PSF array + bead_data dict to a compressed .psfr file."""
        if self._psf is None or self._bead_data is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save results as …",
            defaultextension=".psfr.npz",
            filetypes=[("PSF results", "*.psfr.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        arrays = {"psf": self._psf}
        for key, val in self._bead_data.items():
            if key == "volume_paths":
                arrays["bd_volume_paths"] = np.array([str(p) for p in val])
            elif key == "roi_shape":
                arrays["bd_roi_shape"] = np.array(list(val), dtype=np.int64)
            elif isinstance(val, np.ndarray):
                arrays[f"bd_{key}"] = val
            else:
                arrays[f"bd_{key}"] = np.array(val)
        np.savez_compressed(path, **arrays)
        self.status_var.set(f"Results saved → {os.path.basename(path)}")

    def _load_results(self):
        """Load a .psfr file and restore PSF + bead_data, then refresh all plots."""
        path = filedialog.askopenfilename(
            title="Load results …",
            filetypes=[("PSF results", "*.psfr.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        if "psf" not in data.files:
            messagebox.showerror("Load error", "File does not contain a PSF array.")
            return

        self._psf = data["psf"]

        bd = {}
        for raw_key in data.files:
            if not raw_key.startswith("bd_"):
                continue
            field = raw_key[3:]          # strip "bd_" prefix
            arr   = data[raw_key]
            if field == "volume_paths":
                bd[field] = list(arr.astype(str))
            elif field == "roi_shape":
                bd[field] = tuple(arr.tolist())
            elif arr.ndim == 0:
                bd[field] = arr.item()
            else:
                bd[field] = arr
        self._bead_data     = bd
        self._last_tif_path = None

        self._clear_log()
        self._log_append(f"[Loaded from {os.path.basename(path)}]\n")
        self.clear_btn.config(state="normal")
        self.save_res_btn.config(state="normal")
        self._update_all_plots()
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    # =========================================================================
    # Clear / reset
    # =========================================================================

    def _clear_results(self):
        """Reset the application to its initial state without closing the window.

        Clears all result plots and internal state variables, but leaves all
        input parameters (file paths, pixel sizes, thresholds, etc.) intact so
        the user can re-run with adjusted settings immediately.
        """
        # --- State variables ---
        self._psf           = None
        self._bead_data     = None
        self._fov_cbar      = None
        self._last_tif_path = None

        # --- Progress bar, labels, log, status ---
        self.progress["value"] = 0
        self.pct_lbl.config(text="")
        self._clear_log()
        self.status_var.set("Ready.")
        self.psf_fwhm_var.set("")
        self.psf_theory_var.set("")
        self.beads_stats_var.set("")

        # --- PSF cross-sections tab ---
        for ax, title in zip(
            [self.ax_xy, self.ax_xz, self.ax_yz],
            ["XY  (focal plane)",
             "XZ  (axial · lateral X)",
             "YZ  (axial · lateral Y)"],
        ):
            ax.cla()
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=10)
        self.psf_canvas.draw()

        # --- Beads scatter tab ---
        self.beads_fig.clear()
        self.beads_ax = self.beads_fig.add_subplot(111)
        self.beads_ax.set_title("Bead positions (XY projection)", fontsize=9)
        self.beads_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=self.beads_ax.transAxes, color="gray")
        self.beads_canvas.draw()

        # --- FWHM histograms ---
        for ax, title in zip(
            [self.hist_ax1, self.hist_ax2, self.hist_ax3],
            ["FWHM_xy", "FWHM_z", "FWHM vs bead Z position"],
        ):
            ax.cla()
            ax.set_title(title, fontsize=9)
        self.hist_canvas.draw()

        # --- FOV map tab ---
        self.fov_fig.clear()
        self.fov_ax = self.fov_fig.add_subplot(111)
        self.fov_ax.set_title("PSF variation across the field of view", fontsize=10)
        self.fov_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.fov_ax.transAxes, color="gray")
        self.fov_canvas.draw()

        # --- FWHM diagnostics tab ---
        for ax, title in zip(
            [self.hfit_prof_z, self.hfit_prof_y, self.hfit_prof_x,
             self.hfit_hist_z, self.hfit_hist_y, self.hfit_hist_x],
            ["Profile Z  (avg PSF)", "Profile Y  (avg PSF)", "Profile X  (avg PSF)",
             "FWHM_z  per bead",    "FWHM_y  per bead",    "FWHM_x  per bead"],
        ):
            ax.cla()
            ax.set_title(title, fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        self.hfit_canvas.draw()

        # Disable until the next successful run
        self.clear_btn.config(state="disabled")
        self.save_res_btn.config(state="disabled")

    # =========================================================================
    # Batch merge helpers
    # =========================================================================

    @staticmethod
    def _merge_psfs(results):
        """Weighted average of per-volume PSFs, weighted by n_used beads.

        Each PSF_i is already a nanmean of its n_used_i aligned ROIs,
        normalized to sum=1.  Weighting by n_used_i before averaging is
        mathematically equivalent to computing the nanmean over every bead
        across all volumes simultaneously, which is what the backend would
        produce if all beads came from a single file.
        """
        weights = np.array([bd['n_used'] for _, bd, _ in results], dtype=np.float64)
        if weights.sum() == 0:
            weights = np.ones(len(results), dtype=np.float64)

        psf_stack = np.stack([psf.astype(np.float64) for psf, _, _ in results], axis=0)
        combined  = np.einsum('i,ixyz->xyz', weights, psf_stack) / weights.sum()

        # Re-apply the same normalization the backend uses (min-subtract, sum=1)
        combined -= combined.min()
        s = combined.sum()
        if s > 0:
            combined /= s
        return combined.astype(np.float32)

    @staticmethod
    def _merge_bead_data(results, tif_files):
        """Concatenate per-volume bead_data dicts into one aggregated dict.

        A ``volume_id`` array (int32) is added to every accepted-bead row so
        that scatter plots can colour beads by their source volume.
        ``volume_paths`` preserves the ordered list of source file paths.
        """
        def _vstack_px(arrays):
            """Concatenate (N, 3) pixel arrays, handling empty-list edge cases."""
            nonempty = [a for a in arrays if len(a) > 0]
            if not nonempty:
                return np.zeros((0, 3), dtype=np.int32)
            return np.concatenate(nonempty, axis=0)

        bd0 = results[0][1]

        # Warn once if pixel sizes differ across volumes
        for _, bd, _ in results[1:]:
            if abs(bd['dx'] - bd0['dx']) > 1e-6 or abs(bd['dz'] - bd0['dz']) > 1e-6:
                import warnings
                warnings.warn(
                    "Batch volumes have different pixel sizes; "
                    "using dx/dz from the first volume for display."
                )
                break

        # volume_id: one entry per bead in each category
        vol_id_accepted = [
            np.full(bd['n_accepted'],       vid, dtype=np.int32)
            for vid, (_, bd, _) in enumerate(results)
        ]
        vol_id_border = [
            np.full(len(bd['border_px']),   vid, dtype=np.int32)
            for vid, (_, bd, _) in enumerate(results)
        ]
        vol_id_rejected = [
            np.full(len(bd['rejected_px']), vid, dtype=np.int32)
            for vid, (_, bd, _) in enumerate(results)
        ]

        def _concat_or_empty(arrays):
            nonempty = [a for a in arrays if len(a) > 0]
            return np.concatenate(nonempty) if nonempty else np.array([], dtype=np.int32)

        return {
            'dx':        bd0['dx'],
            'dz':        bd0['dz'],
            'roi_shape': bd0['roi_shape'],
            # Per-bead arrays (accepted)
            'accepted_px':          _vstack_px([r[1]['accepted_px']          for r in results]),
            'accepted_sigma_z':     np.concatenate([r[1]['accepted_sigma_z']     for r in results]),
            'accepted_sigma_y':     np.concatenate([r[1]['accepted_sigma_y']     for r in results]),
            'accepted_sigma_x':     np.concatenate([r[1]['accepted_sigma_x']     for r in results]),
            'accepted_sigma_xy':    np.concatenate([r[1]['accepted_sigma_xy']    for r in results]),
            'accepted_ellipticity': np.concatenate([r[1]['accepted_ellipticity'] for r in results]),
            'accepted_snr':         np.concatenate([r[1]['accepted_snr']         for r in results]),
            'accepted_used':        np.concatenate([r[1]['accepted_used']        for r in results]),
            'volume_id':            _concat_or_empty(vol_id_accepted),
            # Rejected / border arrays
            'border_px':          _vstack_px([r[1]['border_px']   for r in results]),
            'rejected_px':        _vstack_px([r[1]['rejected_px'] for r in results]),
            'border_volume_id':   _concat_or_empty(vol_id_border),
            'rejected_volume_id': _concat_or_empty(vol_id_rejected),
            # Summed counts
            'n_total':            sum(r[1]['n_total']            for r in results),
            'n_border':           sum(r[1]['n_border']           for r in results),
            'n_quality_rejected': sum(r[1]['n_quality_rejected'] for r in results),
            'n_accepted':         sum(r[1]['n_accepted']         for r in results),
            'n_used':             sum(r[1]['n_used']             for r in results),
            # Batch metadata
            'volume_paths': list(tif_files),
            'n_volumes':    len(results),
        }

    @staticmethod
    def _apply_combined_fwhm(psf, bead_data, dz, dx):
        """Compute combined FWHM stats from a merged PSF + merged bead arrays.

        Mutates *bead_data* in-place, injecting the same FWHM keys that
        estimate_psf_from_beads produces for a single volume.  Called by
        _worker after _merge_psfs / _merge_bead_data in batch mode.
        """
        dz_nm = dz * 1000.0
        dx_nm = dx * 1000.0

        avg = measure_fwhm_from_averaged_psf(psf, (dz_nm, dx_nm, dx_nm))
        avg_z, avg_y, avg_x = avg['fwhm_z_nm'], avg['fwhm_y_nm'], avg['fwhm_x_nm']

        used = bead_data['accepted_used']
        fwhm_z = bead_data['accepted_sigma_z'][used] * 2.355 * 1000.0
        fwhm_y = bead_data['accepted_sigma_y'][used] * 2.355 * 1000.0
        fwhm_x = bead_data['accepted_sigma_x'][used] * 2.355 * 1000.0

        def _stats(arr):
            if len(arr) == 0:
                nan = float('nan')
                return nan, nan, nan, nan
            mean_ = float(np.mean(arr))
            sd_   = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            med_  = float(np.median(arr))
            mad_  = float(np.median(np.abs(arr - med_)))
            return mean_, sd_, med_, mad_

        mean_z, sd_z, med_z, mad_z = _stats(fwhm_z)
        mean_y, sd_y, med_y, mad_y = _stats(fwhm_y)
        mean_x, sd_x, med_x, mad_x = _stats(fwhm_x)

        def _r(v):
            return round(v, 1) if np.isfinite(v) else None

        bead_data.update({
            'fwhm_averaged_psf_z': avg_z,
            'fwhm_averaged_psf_y': avg_y,
            'fwhm_averaged_psf_x': avg_x,
            'fwhm_per_bead_mean_z': mean_z, 'fwhm_per_bead_mean_y': mean_y, 'fwhm_per_bead_mean_x': mean_x,
            'fwhm_per_bead_sd_z':   sd_z,   'fwhm_per_bead_sd_y':   sd_y,   'fwhm_per_bead_sd_x':   sd_x,
            'fwhm_median_z': med_z, 'fwhm_median_y': med_y, 'fwhm_median_x': med_x,
            'fwhm_mad_z':    mad_z, 'fwhm_mad_y':    mad_y, 'fwhm_mad_x':    mad_x,
            'reporting_mode': 'combined',
            'fwhm_axes': {
                'axis_z': {'fwhm_averaged_psf_nm': _r(avg_z),
                           'fwhm_per_bead_mean_nm': _r(mean_z), 'fwhm_per_bead_sd_nm': _r(sd_z),
                           'fwhm_per_bead_median_nm': _r(med_z), 'fwhm_per_bead_mad_nm': _r(mad_z)},
                'axis_y': {'fwhm_averaged_psf_nm': _r(avg_y),
                           'fwhm_per_bead_mean_nm': _r(mean_y), 'fwhm_per_bead_sd_nm': _r(sd_y),
                           'fwhm_per_bead_median_nm': _r(med_y), 'fwhm_per_bead_mad_nm': _r(mad_y)},
                'axis_x': {'fwhm_averaged_psf_nm': _r(avg_x),
                           'fwhm_per_bead_mean_nm': _r(mean_x), 'fwhm_per_bead_sd_nm': _r(sd_x),
                           'fwhm_per_bead_median_nm': _r(med_x), 'fwhm_per_bead_mad_nm': _r(mad_x)},
            },
        })

    # =========================================================================
    # Plot updates
    # =========================================================================

    def _update_all_plots(self):
        if self._psf is not None:
            self._update_psf_plot()
        if self._bead_data is not None:
            self._update_beads_plot()
            self._refresh_fov()
            self._update_hist_fit_plots()
        self.nb.select(1)   # switch to PSF tab

    # ── PSF cross-sections ────────────────────────────────────────────────────

    def _update_psf_plot(self):
        psf = self._psf
        nz, ny, nx = psf.shape
        cz, cy, cx = nz // 2, ny // 2, nx // 2

        try:
            dx = float(self.dx_var.get())
            dz = float(self.dz_var.get())
        except ValueError:
            dx = dz = 1.0

        ext_xy = [0, nx * dx, 0, ny * dx]
        ext_xz = [0, nx * dx, 0, nz * dz]
        ext_yz = [0, ny * dx, 0, nz * dz]

        for ax in [self.ax_xy, self.ax_xz, self.ax_yz]:
            ax.cla()

        self.ax_xy.imshow(psf[cz],       origin="lower", cmap="hot",
                          extent=ext_xy,  aspect="equal")
        self.ax_xz.imshow(psf[:, cy, :], origin="lower", cmap="hot",
                          extent=ext_xz,  aspect="auto")
        self.ax_yz.imshow(psf[:, :, cx], origin="lower", cmap="hot",
                          extent=ext_yz,  aspect="auto")

        for ax, title, xl, yl, ch, cv in [
            (self.ax_xy, "XY  (focal plane)",       "x (µm)", "y (µm)", cy*dx, cx*dx),
            (self.ax_xz, "XZ  (axial · lateral X)", "x (µm)", "z (µm)", cz*dz, cx*dx),
            (self.ax_yz, "YZ  (axial · lateral Y)", "y (µm)", "z (µm)", cz*dz, cy*dx),
        ]:
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.axhline(ch, color="cyan", lw=0.6, alpha=0.8)
            ax.axvline(cv, color="cyan", lw=0.6, alpha=0.8)

        self.psf_fwhm_var.set(self._compute_fwhm_str(psf, dx, dz))

        theory = self._get_theory_fwhm()
        if theory:
            fxy_t, fz_t = theory
            self.psf_theory_var.set(
                f"Theory (Born-Wolf):   FWHM_xy = {fxy_t:.0f} nm"
                f"   |   FWHM_z = {fz_t:.0f} nm"
            )
        else:
            self.psf_theory_var.set("")

        self.psf_fig.tight_layout()
        self.psf_canvas.draw()

    def _save_psf_projections(self):
        if self._psf is None:
            messagebox.showwarning("No data", "Run PSF estimation first.")
            return
        from tifffile import imwrite as _imwrite
        from scipy.ndimage import zoom as _zoom
        psf = self._psf
        nz, ny, nx = psf.shape
        cz, cy, cx = nz // 2, ny // 2, nx // 2

        try:
            dx = float(self.dx_var.get())
            dz = float(self.dz_var.get())
        except ValueError:
            dx = dz = 1.0

        base = filedialog.asksaveasfilename(
            title="Save projections — choose base name (suffix _XY/XZ/YZ added)",
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif *.tiff")],
        )
        if not base:
            return

        root, ext = os.path.splitext(base)
        if ext.lower() not in (".tif", ".tiff"):
            ext = ".tif"

        def _to_uint16(arr):
            a = arr.astype(np.float32)
            mn, mx = a.min(), a.max()
            if mx > mn:
                a = (a - mn) / (mx - mn)
            return (a * 65535).astype(np.uint16)

        # XZ and YZ: resample z-axis so every pixel = dx µm (isotropic)
        z_factor = dz / dx
        xy_slice = psf[cz]
        xz_slice = _zoom(psf[:, cy, :], zoom=(z_factor, 1.0), order=3)
        yz_slice = _zoom(psf[:, :, cx], zoom=(z_factor, 1.0), order=3)

        # ImageJ-compatible metadata: resolution in pixels-per-µm
        res = (1.0 / dx, 1.0 / dx)   # isotropic after zoom
        ij_meta = {"unit": "um"}

        projections = {
            "XY": xy_slice,
            "XZ": xz_slice,
            "YZ": yz_slice,
        }
        saved = []
        for label, arr in projections.items():
            path = f"{root}_{label}{ext}"
            _imwrite(path, _to_uint16(arr),
                     imagej=True, resolution=res, metadata=ij_meta)
            saved.append(os.path.basename(path))

        messagebox.showinfo("Saved", "Projections saved:\n" + "\n".join(saved))

    def _compute_fwhm_str(self, psf, dx, dz):
        """Measure FWHM from 1-D central profiles of the averaged PSF (non-parametric)."""
        from postprocess_psf import measure_fwhm_from_averaged_psf
        try:
            fwhms = measure_fwhm_from_averaged_psf(
                psf, (dz * 1000.0, dx * 1000.0, dx * 1000.0)
            )
        except Exception:
            return ""
        fz = fwhms['fwhm_z_nm']
        fy = fwhms['fwhm_y_nm']
        fx = fwhms['fwhm_x_nm']
        parts = []
        if np.isfinite(fz): parts.append(f"FWHM_z = {fz:.0f} nm")
        if np.isfinite(fy): parts.append(f"FWHM_y = {fy:.0f} nm")
        if np.isfinite(fx): parts.append(f"FWHM_x = {fx:.0f} nm")
        if np.isfinite(fy) and np.isfinite(fx):
            parts.append(f"FWHM_xy = {(fy + fx) / 2:.0f} nm")
        return "   |   ".join(parts)

    # ── Beads: scatter + histograms ───────────────────────────────────────────

    def _update_beads_plot(self):
        bd = self._bead_data
        dx = bd['dx']

        # Clear the whole figure so the colorbar axes don't accumulate.
        self.beads_fig.clear()
        self.beads_ax = self.beads_fig.add_subplot(111)

        # Border-rejected beads (light gray ×)
        bp = bd['border_px']
        if len(bp):
            self.beads_ax.scatter(bp[:, 2] * dx, bp[:, 1] * dx,
                                  marker="x", s=18, color="lightgray", alpha=0.5,
                                  label=f"Border ({len(bp)})", zorder=1)

        # Quality-rejected beads (salmon ×)
        rp = bd['rejected_px']
        if len(rp):
            self.beads_ax.scatter(rp[:, 2] * dx, rp[:, 1] * dx,
                                  marker="x", s=20, color="salmon", alpha=0.7,
                                  label=f"Quality reject ({len(rp)})", zorder=2)

        acc_px  = bd['accepted_px']
        used    = bd['accepted_used']
        sxy_all = bd['accepted_sigma_xy']

        is_batch  = 'volume_id' in bd
        not_used_idx = np.where(~used)[0]
        used_idx     = np.where( used)[0]

        if is_batch:
            # ── Batch mode: colour accepted beads by source volume ────────────
            n_vols     = bd['n_volumes']
            vol_id     = bd['volume_id']
            vol_paths  = bd.get('volume_paths', [])
            cmap_vols  = matplotlib.cm.tab10
            vol_colors = [cmap_vols(i % 10 / 10) for i in range(n_vols)]

            for vid in range(n_vols):
                vol_mask = vol_id == vid
                col      = vol_colors[vid]
                vol_name = os.path.basename(vol_paths[vid]) if vid < len(vol_paths) else f"Vol {vid}"

                nu_vol = np.where(~used & vol_mask)[0]
                u_vol  = np.where( used & vol_mask)[0]

                if len(nu_vol):
                    nup = acc_px[nu_vol]
                    self.beads_ax.scatter(nup[:, 2] * dx, nup[:, 1] * dx,
                                          marker="o", s=22,
                                          facecolors="none", edgecolors=col,
                                          linewidths=0.9, alpha=0.7,
                                          zorder=3)
                if len(u_vol):
                    up = acc_px[u_vol]
                    self.beads_ax.scatter(up[:, 2] * dx, up[:, 1] * dx,
                                          marker="o", s=55, color=col,
                                          edgecolors="black", lw=0.4, alpha=0.85,
                                          label=f"{vol_name} ({len(u_vol)} used)",
                                          zorder=5)

            scatter_title = (f"Bead positions — {n_vols} volumes  "
                             f"(colour = volume,  ○ outline = not used)")
        else:
            # ── Single file: colour used beads by σ_xy ────────────────────────
            if len(not_used_idx):
                nup = acc_px[not_used_idx]
                self.beads_ax.scatter(nup[:, 2] * dx, nup[:, 1] * dx,
                                      marker="o", s=28, color="steelblue", alpha=0.55,
                                      label=f"Accepted, not used ({len(not_used_idx)})",
                                      zorder=3)

            if len(used_idx):
                up = acc_px[used_idx]
                us = sxy_all[used_idx] * 1000   # nm
                sc = self.beads_ax.scatter(up[:, 2] * dx, up[:, 1] * dx,
                                           c=us, cmap="viridis_r",
                                           vmin=us.min(), vmax=us.max(),
                                           marker="o", s=60,
                                           edgecolors="black", lw=0.4,
                                           label=f"Used in PSF ({len(used_idx)})",
                                           zorder=5)
                self.beads_fig.colorbar(sc, ax=self.beads_ax,
                                        label="σ_xy (nm)", shrink=0.85)

            scatter_title = "Bead positions (XY projection)"

        self.beads_ax.set_xlabel("x (µm)", fontsize=9)
        self.beads_ax.set_ylabel("y (µm)", fontsize=9)
        self.beads_ax.set_title(scatter_title, fontsize=10)
        self.beads_ax.legend(fontsize=7, loc="upper right")
        self.beads_ax.set_aspect("equal")

        # Histograms + FWHM vs Z scatter
        self.hist_ax1.cla()
        self.hist_ax2.cla()
        self.hist_ax3.cla()

        fwhm_xy = sxy_all * 2.355 * 1000                 # nm
        fwhm_z  = bd['accepted_sigma_z'] * 2.355 * 1000  # nm
        bead_z  = acc_px[:, 0] * bd['dz']                # axial position in µm

        if len(used_idx):
            self.hist_ax1.hist(fwhm_xy[used_idx], bins=15, color="green",
                               alpha=0.75, label="Used")
            self.hist_ax2.hist(fwhm_z[used_idx],  bins=15, color="green",
                               alpha=0.75, label="Used")
        if len(not_used_idx):
            self.hist_ax1.hist(fwhm_xy[not_used_idx], bins=15, color="steelblue",
                               alpha=0.5, label="Not used")
            self.hist_ax2.hist(fwhm_z[not_used_idx],  bins=15, color="steelblue",
                               alpha=0.5, label="Not used")

        # Theoretical reference lines (dashed orange)
        theory = self._get_theory_fwhm()
        if theory:
            fxy_t, fz_t = theory
            self.hist_ax1.axvline(fxy_t, color="darkorange", lw=1.5, ls="--",
                                  label=f"Theory {fxy_t:.0f} nm")
            self.hist_ax2.axvline(fz_t,  color="darkorange", lw=1.5, ls="--",
                                  label=f"Theory {fz_t:.0f} nm")

        for ax, lbl in [(self.hist_ax1, "FWHM_xy (nm)"),
                        (self.hist_ax2, "FWHM_z (nm)")]:
            ax.set_xlabel(lbl, fontsize=8)
            ax.legend(fontsize=7)

        # FWHM vs bead Z position (detects depth-dependent aberrations)
        if is_batch:
            for vid in range(bd['n_volumes']):
                vol_mask = bd['volume_id'] == vid
                col      = vol_colors[vid]
                nu_vol = np.where(~used & vol_mask)[0]
                u_vol  = np.where( used & vol_mask)[0]
                if len(nu_vol):
                    self.hist_ax3.scatter(bead_z[nu_vol], fwhm_xy[nu_vol],
                                          s=18, alpha=0.45, marker="o",
                                          facecolors="none", edgecolors=col)
                if len(u_vol):
                    vol_name = (os.path.basename(bd['volume_paths'][vid])
                                if vid < len(bd.get('volume_paths', [])) else f"Vol {vid}")
                    self.hist_ax3.scatter(bead_z[u_vol], fwhm_xy[u_vol],
                                          color=col, s=28, alpha=0.8,
                                          label=vol_name)
        else:
            if len(not_used_idx):
                self.hist_ax3.scatter(bead_z[not_used_idx], fwhm_xy[not_used_idx],
                                      c="steelblue", s=22, alpha=0.55, label="Not used")
            if len(used_idx):
                self.hist_ax3.scatter(bead_z[used_idx], fwhm_xy[used_idx],
                                      c="green", s=30, alpha=0.8, label="Used")
        if theory:
            fxy_t, _ = theory
            self.hist_ax3.axhline(fxy_t, color="darkorange", lw=1.5, ls="--",
                                  label=f"Theory {fxy_t:.0f} nm")
        self.hist_ax3.set_xlabel("Bead Z position (µm)", fontsize=8)
        self.hist_ax3.set_ylabel("FWHM_xy (nm)", fontsize=8)
        self.hist_ax3.set_title("FWHM_xy vs bead Z position", fontsize=9)
        self.hist_ax3.legend(fontsize=7)

        # Summary statistics — filter funnel
        n_used = len(used_idx)
        vol_prefix = f"Volumes: {bd['n_volumes']}  |  " if is_batch else ""

        # New-style filter counts (may not exist in files saved by older versions)
        if 'n_edge' in bd:
            funnel = (f"detected={bd['n_total']} → edge={bd['n_edge']} → "
                      f"isolation={bd['n_isolation']} → fit_ok={bd['n_fit_ok']} → "
                      f"amplitude={bd['n_amplitude']} → r²={bd['n_r2']} → "
                      f"sanity={bd['n_sanity']}")
        else:
            funnel = (f"Total={bd['n_total']}  border={bd['n_border']}  "
                      f"quality_rejected={bd['n_quality_rejected']}  "
                      f"accepted={bd['n_accepted']}")

        stats = f"{vol_prefix}Beads: {funnel}  |  Used: {n_used}"

        if n_used:
            avg_z = bd.get('fwhm_averaged_psf_z', float('nan'))
            avg_y = bd.get('fwhm_averaged_psf_y', float('nan'))
            avg_x = bd.get('fwhm_averaged_psf_x', float('nan'))
            sd_z  = bd.get('fwhm_per_bead_sd_z',  float('nan'))
            med_y = bd.get('fwhm_median_y',        float('nan'))
            med_x = bd.get('fwhm_median_x',        float('nan'))

            if np.isfinite(avg_z):
                avg_xy = 0.5 * (avg_y + avg_x) if (np.isfinite(avg_y) and np.isfinite(avg_x)) \
                         else float('nan')
                med_xy = 0.5 * (med_y + med_x)
                avg_xy_s = f"{avg_xy:.0f}" if np.isfinite(avg_xy) else "?"
                sd_z_s   = f" ±{sd_z:.0f} SD" if np.isfinite(sd_z) else ""
                stats += (f"\nFWHM_xy: {avg_xy_s} nm (avg-PSF)   "
                          f"per-bead median: {med_xy:.0f} nm"
                          f"\nFWHM_z:  {avg_z:.0f} nm (avg-PSF){sd_z_s} per-bead")
            else:
                stats += (f"\nFWHM_xy: {fwhm_xy[used_idx].mean():.0f} ± "
                          f"{fwhm_xy[used_idx].std():.0f} nm    "
                          f"FWHM_z: {fwhm_z[used_idx].mean():.0f} ± "
                          f"{fwhm_z[used_idx].std():.0f} nm")
        self.beads_stats_var.set(stats)

        self.beads_fig.tight_layout()
        self.beads_canvas.draw()
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()

    # ── FWHM histogram fit plots ──────────────────────────────────────────────

    def _update_hist_fit_plots(self):
        bd  = self._bead_data
        psf = self._psf
        if bd is None or psf is None:
            return

        from postprocess_psf import measure_fwhm_from_averaged_psf
        from scipy.interpolate import CubicSpline

        try:
            dx = float(self.dx_var.get())
            dz = float(self.dz_var.get())
        except ValueError:
            dx = dz = 1.0

        # ── Top row: 1-D profiles of the averaged PSF ─────────────────────────
        try:
            avg_fwhm = measure_fwhm_from_averaged_psf(
                psf, (dz * 1000.0, dx * 1000.0, dx * 1000.0)
            )
        except Exception:
            avg_fwhm = {'fwhm_z_nm': float('nan'),
                        'fwhm_y_nm': float('nan'),
                        'fwhm_x_nm': float('nan')}

        nz, ny, nx = psf.shape
        iz, iy, ix = np.unravel_index(np.argmax(psf), psf.shape)

        profile_specs = [
            (self.hfit_prof_z, psf[:, iy, ix], np.arange(nz) * dz * 1000,
             "z (nm)", "Profile Z  (avg PSF)", avg_fwhm['fwhm_z_nm']),
            (self.hfit_prof_y, psf[iz, :, ix], np.arange(ny) * dx * 1000,
             "y (nm)", "Profile Y  (avg PSF)", avg_fwhm['fwhm_y_nm']),
            (self.hfit_prof_x, psf[iz, iy, :], np.arange(nx) * dx * 1000,
             "x (nm)", "Profile X  (avg PSF)", avg_fwhm['fwhm_x_nm']),
        ]

        for ax, profile, coords, xlabel, title, fwhm_nm in profile_specs:
            ax.cla()
            ax.set_title(title, fontsize=9)

            bg       = float(np.percentile(profile, 5))
            peak     = float(np.max(profile)) - bg
            half_max = bg + 0.5 * peak if peak > 0 else float('nan')

            coords_fine  = np.linspace(coords[0], coords[-1], len(coords) * 10)
            profile_fine = CubicSpline(coords, profile)(coords_fine)

            ax.plot(coords, profile, 'o', ms=3, color='steelblue', alpha=0.6,
                    zorder=3)
            ax.plot(coords_fine, profile_fine, '-', color='steelblue', lw=1.2,
                    zorder=2)
            if np.isfinite(half_max):
                ax.axhline(half_max, color='red', lw=0.9, ls='--', alpha=0.8,
                           label="50%")
            ax.axhline(bg, color='gray', lw=0.6, ls=':', alpha=0.6,
                       label="BG (p5)")

            if np.isfinite(fwhm_nm):
                ax.text(0.05, 0.92, f"FWHM = {fwhm_nm:.0f} nm",
                        transform=ax.transAxes, fontsize=8, color='red',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")

        # ── Bottom row: per-bead FWHM histograms ──────────────────────────────
        used      = bd['accepted_used']
        fwhm_z_nm = bd['accepted_sigma_z'][used] * 2.355 * 1000
        fwhm_y_nm = bd['accepted_sigma_y'][used] * 2.355 * 1000
        fwhm_x_nm = bd['accepted_sigma_x'][used] * 2.355 * 1000

        hist_specs = [
            (self.hfit_hist_z, fwhm_z_nm, "FWHM_z (nm)", "FWHM_z  per bead",
             bd.get('fwhm_per_bead_mean_z'), bd.get('fwhm_per_bead_sd_z'),
             bd.get('fwhm_median_z'),         bd.get('fwhm_mad_z')),
            (self.hfit_hist_y, fwhm_y_nm, "FWHM_y (nm)", "FWHM_y  per bead",
             bd.get('fwhm_per_bead_mean_y'), bd.get('fwhm_per_bead_sd_y'),
             bd.get('fwhm_median_y'),         bd.get('fwhm_mad_y')),
            (self.hfit_hist_x, fwhm_x_nm, "FWHM_x (nm)", "FWHM_x  per bead",
             bd.get('fwhm_per_bead_mean_x'), bd.get('fwhm_per_bead_sd_x'),
             bd.get('fwhm_median_x'),         bd.get('fwhm_mad_x')),
        ]

        for ax, fwhm_nm, xlabel, title, mn, sd, med, mad in hist_specs:
            ax.cla()
            ax.set_title(title, fontsize=9)

            if len(fwhm_nm) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xlabel(xlabel, fontsize=8)
                continue

            counts, edges = np.histogram(fwhm_nm, bins='fd')
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(centers, counts, width=np.diff(edges),
                   color="steelblue", alpha=0.50)

            # Fallback to raw arrays if bead_data scalars not available
            mn_v  = mn  if mn  is not None else float(np.mean(fwhm_nm))
            sd_v  = sd  if sd  is not None else float(np.std(fwhm_nm))
            med_v = med if med is not None else float(np.median(fwhm_nm))
            mad_v = mad if mad is not None else float(
                np.median(np.abs(fwhm_nm - np.median(fwhm_nm))))

            ax.axvline(mn_v,  color="steelblue", lw=1.8,
                       label=f"Mean   {mn_v:.0f} nm")
            ax.axvline(med_v, color="green",     lw=1.8, ls="--",
                       label=f"Median {med_v:.0f} nm")
            ax.axvspan(mn_v  - sd_v,  mn_v  + sd_v,  alpha=0.10,
                       color="steelblue", label=f"±SD {sd_v:.0f}")
            ax.axvspan(med_v - mad_v, med_v + mad_v, alpha=0.10,
                       color="green",     label=f"±MAD {mad_v:.0f}")

            txt = (f"Mean   = {mn_v:.0f} ± {sd_v:.0f} nm\n"
                   f"Median = {med_v:.0f} ± {mad_v:.0f} nm\n"
                   f"N = {len(fwhm_nm)}")
            ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")

        self.hfit_fig.tight_layout()
        self.hfit_canvas.draw()

    # ── FOV resolution map ────────────────────────────────────────────────────

    def _refresh_fov(self):
        bd = self._bead_data
        if bd is None:
            return

        dx       = bd['dx']
        metric   = self.fov_metric.get()
        show_all = self.fov_all_var.get()

        used = bd['accepted_used']
        mask = np.ones(len(used), dtype=bool) if show_all else used

        acc_px = bd['accepted_px']
        is_ellipticity = (metric == "Ellipticity")
        metric_map = {
            "FWHM_xy":    (bd['accepted_sigma_xy'] * 2.355 * 1000, "RdYlGn_r", "nm"),
            "FWHM_z":     (bd['accepted_sigma_z']  * 2.355 * 1000, "RdYlGn_r", "nm"),
            "FWHM_y":     (bd['accepted_sigma_y']  * 2.355 * 1000, "RdYlGn_r", "nm"),
            "FWHM_x":     (bd['accepted_sigma_x']  * 2.355 * 1000, "RdYlGn_r", "nm"),
            "Ellipticity":(bd['accepted_ellipticity'],              "RdBu_r",   ""),
            "SNR":        (bd['accepted_snr'],                      "plasma",   ""),
        }
        values, cmap_name, unit = metric_map[metric]

        # Clear the entire figure and recreate the axes so that colorbars
        # (which live in their own axes) do not accumulate across refreshes.
        self.fov_fig.clear()
        self.fov_ax   = self.fov_fig.add_subplot(111)
        self._fov_cbar = None

        xs = acc_px[mask, 2] * dx   # x in µm
        ys = acc_px[mask, 1] * dx   # y in µm
        vs = values[mask]

        if len(xs) == 0:
            self.fov_ax.text(0.5, 0.5, "No beads to display",
                             ha="center", va="center",
                             transform=self.fov_ax.transAxes, color="gray")
        else:
            # For ellipticity use a symmetric diverging scale centred at 0
            if is_ellipticity:
                half = max(abs(vs.min()), abs(vs.max()))
                vmin_c, vmax_c = -half, half
            else:
                vmin_c, vmax_c = vs.min(), vs.max()

            sc = self.fov_ax.scatter(
                xs, ys, c=vs, cmap=cmap_name,
                vmin=vmin_c, vmax=vmax_c,
                s=90, edgecolors="black", lw=0.4, zorder=3,
            )
            cbar_label = metric if not unit else f"{metric} ({unit})"
            self._fov_cbar = self.fov_fig.colorbar(
                sc, ax=self.fov_ax, label=cbar_label, shrink=0.85
            )

            # Annotate global minimum and maximum
            if len(vs) > 1:
                i_min, i_max = np.argmin(vs), np.argmax(vs)
                is_snr = (metric == "SNR")
                if is_ellipticity:
                    ann_pairs = [
                        (i_min, "steelblue", f"min\n{vs[i_min]:.3f}"),
                        (i_max, "firebrick", f"max\n{vs[i_max]:.3f}"),
                    ]
                elif is_snr:
                    # High SNR = good (max is green, min is red)
                    ann_pairs = [
                        (i_min, "darkred",   f"min\n{vs[i_min]:.1f}"),
                        (i_max, "darkgreen", f"max\n{vs[i_max]:.1f}"),
                    ]
                else:
                    # FWHM: low = good (min is green, max is red)
                    ann_pairs = [
                        (i_min, "darkgreen", f"min\n{vs[i_min]:.0f} nm"),
                        (i_max, "darkred",   f"max\n{vs[i_max]:.0f} nm"),
                    ]
                for idx, color, label in ann_pairs:
                    self.fov_ax.annotate(
                        label, (xs[idx], ys[idx]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                    )

            # Title encodes range information
            subset = "all accepted" if show_all else "used in PSF"
            is_snr = (metric == "SNR")
            if is_ellipticity:
                rng = vs.max() - vs.min()
                title_range = (f"range: {vs.min():.3f}–{vs.max():.3f}  "
                               f"(Δ {rng:.3f})")
            elif is_snr:
                rng = vs.max() - vs.min()
                title_range = (f"range: {vs.min():.1f}–{vs.max():.1f}  "
                               f"(Δ {rng:.1f})")
            else:
                rng = vs.max() - vs.min()
                title_range = (f"range: {vs.min():.0f}–{vs.max():.0f} nm  "
                               f"(Δ {rng:.0f} nm)")
            self.fov_ax.set_title(
                f"{metric} variation across the FOV  ({subset})  —  {title_range}",
                fontsize=9,
            )

        self.fov_ax.set_xlabel("x (µm)", fontsize=9)
        self.fov_ax.set_ylabel("y (µm)", fontsize=9)
        self.fov_ax.set_aspect("equal")
        self.fov_fig.tight_layout()
        self.fov_canvas.draw()

    # =========================================================================
    # Theoretical PSF helpers
    # =========================================================================

    def _get_theory_fwhm(self):
        """Return (fwhm_xy_nm, fwhm_z_nm) from Born-Wolf model, or None.

        Formulas
        --------
        FWHM_xy = 0.51 · λ / NA
        FWHM_z  = 0.887 · λ / (n − √(n² − NA²))
        where λ is in nm and all inputs are in consistent units.
        """
        if not self.show_theory_var.get():
            return None
        try:
            lam = float(self.lambda_var.get())
            na  = float(self.na_var.get())
            n   = float(self.n_var.get())
            if lam <= 0 or na <= 0 or n <= 0 or na >= n:
                return None
            fwhm_xy = 0.51 * lam / na
            fwhm_z  = 0.887 * lam / (n - np.sqrt(n**2 - na**2))
            return fwhm_xy, fwhm_z
        except (ValueError, ZeroDivisionError):
            return None

    def _refresh_theory_overlay(self):
        """Redraw active plots to add or remove theoretical overlay lines."""
        if self._psf is not None:
            self._update_psf_plot()
        if self._bead_data is not None:
            self._update_beads_plot()

    # =========================================================================
    # Click-to-inspect
    # =========================================================================

    def _on_beads_click(self, event):
        """Left-click on the bead scatter to open a per-bead inspector popup."""
        if self._bead_data is None:
            return
        if event.inaxes is not self.beads_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        # Don't fire while the toolbar is in pan/zoom mode
        if hasattr(self, 'beads_toolbar') and self.beads_toolbar.mode != '':
            self.status_var.set(
                "Deactivate the zoom/pan tool (toolbar) to use click-to-inspect."
            )
            return

        bd = self._bead_data
        dx = bd['dx']

        # Build a flat list of all beads:
        # (x_um, y_um, z_px, y_px, x_px, category, accepted_idx)
        beads = []
        for bz, by, bx in bd['border_px']:
            beads.append((float(bx)*dx, float(by)*dx,
                          int(bz), int(by), int(bx), 'border_rejected', None))
        for bz, by, bx in bd['rejected_px']:
            beads.append((float(bx)*dx, float(by)*dx,
                          int(bz), int(by), int(bx), 'quality_rejected', None))
        for i, (bz, by, bx) in enumerate(bd['accepted_px']):
            st = 'used_in_psf' if bd['accepted_used'][i] else 'accepted'
            beads.append((float(bx)*dx, float(by)*dx,
                          int(bz), int(by), int(bx), st, i))

        if not beads:
            return

        cx_c, cy_c = event.xdata, event.ydata
        dists = [(b[0] - cx_c) ** 2 + (b[1] - cy_c) ** 2 for b in beads]
        self._show_bead_popup(beads[int(np.argmin(dists))])

    def _show_bead_popup(self, bead_info):
        """Open a Toplevel window showing the ROI cross-sections and 1D profiles."""
        x_um, y_um, z_px, y_px, x_px, category, acc_idx = bead_info
        bd  = self._bead_data
        dx  = bd['dx']
        dz  = bd['dz']

        # Load the source volume
        if not self._last_tif_path or not os.path.isfile(self._last_tif_path):
            messagebox.showwarning(
                "Volume unavailable",
                "The source TIFF is not accessible.\n"
                "Bead inspection requires the original volume file.",
            )
            return
        try:
            from tifffile import imread as _imread
            volume = _imread(self._last_tif_path).astype(np.float32)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        roi_shape = bd.get('roi_shape', (25, 41, 41))
        rz = roi_shape[0] // 2
        ry = roi_shape[1] // 2
        rx = roi_shape[2] // 2
        nz, ny, nx = volume.shape

        # Extract ROI with zero-padding for border beads
        z0, z1 = z_px - rz, z_px + rz + 1
        y0, y1 = y_px - ry, y_px + ry + 1
        x0, x1 = x_px - rx, x_px + rx + 1

        roi = np.zeros(roi_shape, dtype=np.float32)
        vz0, vz1 = max(0, z0), min(nz, z1)
        vy0, vy1 = max(0, y0), min(ny, y1)
        vx0, vx1 = max(0, x0), min(nx, x1)
        if vz1 > vz0 and vy1 > vy0 and vx1 > vx0:
            roi[vz0 - z0: vz0 - z0 + (vz1 - vz0),
                vy0 - y0: vy0 - y0 + (vy1 - vy0),
                vx0 - x0: vx0 - x0 + (vx1 - vx0)] = volume[vz0:vz1, vy0:vy1, vx0:vx1]

        roi -= float(np.percentile(roi, 5))
        roi  = np.clip(roi, 0.0, None)

        nz_r, ny_r, nx_r = roi.shape
        # Use the intensity peak for slicing and profiles
        iz_pk, iy_pk, ix_pk = np.unravel_index(np.argmax(roi), roi.shape)

        z_c = np.arange(nz_r) * dz
        y_c = np.arange(ny_r) * dx
        x_c = np.arange(nx_r) * dx

        # --- Toplevel window ---
        win = tk.Toplevel(self.root)
        status_str = category.replace('_', ' ')
        win.title(
            f"Bead inspector  |  {status_str}  |  "
            f"x={x_um:.1f}  y={y_um:.1f}  z={z_px*dz:.1f} µm"
        )
        win.resizable(True, True)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        # Ensure the popup appears in front of the main window (needed on Windows)
        win.transient(self.root)
        win.lift()
        win.focus_force()

        fig = Figure(figsize=(11, 7), tight_layout=True)

        # --- Row 1: cross-section images ---
        ax_xy = fig.add_subplot(231)
        ax_xz = fig.add_subplot(232)
        ax_yz = fig.add_subplot(233)

        ext_xy = [0, nx_r * dx, 0, ny_r * dx]
        ext_xz = [0, nx_r * dx, 0, nz_r * dz]
        ext_yz = [0, ny_r * dx, 0, nz_r * dz]

        ax_xy.imshow(roi[iz_pk],         origin="lower", cmap="hot",
                     extent=ext_xy, aspect="equal")
        ax_xz.imshow(roi[:, iy_pk, :],   origin="lower", cmap="hot",
                     extent=ext_xz, aspect="auto")
        ax_yz.imshow(roi[:, :, ix_pk],   origin="lower", cmap="hot",
                     extent=ext_yz, aspect="auto")

        for ax, title, xl, yl, ch, cv in [
            (ax_xy, "XY  (focal plane)", "x (µm)", "y (µm)", iy_pk*dx, ix_pk*dx),
            (ax_xz, "XZ  (axial · X)",   "x (µm)", "z (µm)", iz_pk*dz, ix_pk*dx),
            (ax_yz, "YZ  (axial · Y)",   "y (µm)", "z (µm)", iz_pk*dz, iy_pk*dx),
        ]:
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.axhline(ch, color="cyan", lw=0.6, alpha=0.8)
            ax.axvline(cv, color="cyan", lw=0.6, alpha=0.8)

        # --- Row 2: 1D profiles with Gaussian fit ---
        ax_pz = fig.add_subplot(234)
        ax_py = fig.add_subplot(235)
        ax_px = fig.add_subplot(236)

        sz_val = float(bd['accepted_sigma_z'][acc_idx])   if acc_idx is not None else None
        sy_val = float(bd['accepted_sigma_y'][acc_idx])   if acc_idx is not None else None
        sx_val = float(bd['accepted_sigma_x'][acc_idx])   if acc_idx is not None else None

        def _plot_profile(ax, coords, profile, x_label, title, sigma_um):
            ax.plot(coords, profile, 'b.-', ms=4, lw=1, label="Data")
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            if sigma_um is not None:
                bg0 = float(np.percentile(profile, 10))
                A0  = float(np.max(profile)) - bg0
                c0  = coords[np.argmax(profile)]
                try:
                    popt, _ = curve_fit(
                        lambda x, A, c, s, bg: A * np.exp(-(x - c) ** 2 / (2 * s ** 2)) + bg,
                        coords, profile.astype(float),
                        p0=[A0, c0, sigma_um, bg0],
                        bounds=([0, coords[0], 1e-6, 0],
                                [np.inf, coords[-1], np.inf, np.inf]),
                        maxfev=2000,
                    )
                    fine = np.linspace(coords[0], coords[-1], 300)
                    fwhm_nm = abs(popt[2]) * 2355
                    ax.plot(fine,
                            popt[0] * np.exp(-(fine - popt[1]) ** 2 / (2 * popt[2] ** 2)) + popt[3],
                            'r-', lw=1.5, label=f"Fit  FWHM={fwhm_nm:.0f} nm")
                except Exception:
                    pass
            ax.legend(fontsize=7)

        _plot_profile(ax_pz, z_c, roi[:, iy_pk, ix_pk], "Z (µm)", "Z profile", sz_val)
        _plot_profile(ax_py, y_c, roi[iz_pk, :, ix_pk], "Y (µm)", "Y profile", sy_val)
        _plot_profile(ax_px, x_c, roi[iz_pk, iy_pk, :], "X (µm)", "X profile", sx_val)

        # --- Embed figure ---
        canvas_pop = FigureCanvasTkAgg(fig, master=win)
        canvas_pop.draw()
        canvas_pop.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        tb_f = ttk.Frame(win)
        tb_f.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(canvas_pop, tb_f)

        # --- Info text ---
        info_f = ttk.Frame(win)
        info_f.grid(row=2, column=0, sticky="ew", padx=10, pady=4)

        lines = [
            f"Status:    {status_str}",
            f"Position:  x={x_um:.2f} µm   y={y_um:.2f} µm   z={z_px*dz:.2f} µm"
            f"   ({x_px}, {y_px}, {z_px} px)",
        ]
        if acc_idx is not None:
            sz_v  = bd['accepted_sigma_z'][acc_idx]
            sy_v  = bd['accepted_sigma_y'][acc_idx]
            sx_v  = bd['accepted_sigma_x'][acc_idx]
            sxy_v = bd['accepted_sigma_xy'][acc_idx]
            ell_v = bd['accepted_ellipticity'][acc_idx]
            snr_v = bd['accepted_snr'][acc_idx]
            lines += [
                f"σ_z  = {sz_v:.3f} µm   FWHM_z  = {sz_v*2355:.0f} nm",
                f"σ_y  = {sy_v:.3f} µm   FWHM_y  = {sy_v*2355:.0f} nm",
                f"σ_x  = {sx_v:.3f} µm   FWHM_x  = {sx_v*2355:.0f} nm",
                f"σ_xy = {sxy_v:.3f} µm   FWHM_xy = {sxy_v*2355:.0f} nm",
                f"Ellipticity = {ell_v:.4f}",
                f"SNR         = {snr_v:.2f}",
            ]

        ttk.Label(info_f, text="\n".join(lines),
                  font=("Consolas", 9), justify="left",
                  foreground="#1a4080").grid(row=0, column=0, sticky="w")

        ttk.Button(win, text="Close", command=win.destroy).grid(
            row=3, column=0, pady=6)

    # =========================================================================
    # CSV export
    # =========================================================================

    def _export_csv(self):
        """Write a per-bead table to a CSV file chosen by the user.

        Columns
        -------
        bead_id                     — sequential integer (all categories)
        volume_id                   — 0-based source volume index (batch only)
        source_file                 — basename of the source TIFF (batch only)
        z_px, y_px, x_px            — position in pixels
        z_um, y_um, x_um            — position in µm
        status                      — border_rejected | quality_rejected |
                                      accepted | used_in_psf
        sigma_z/y/x/xy_um           — Gaussian sigma in µm  (blank if rejected)
        fwhm_z/y/x/xy_nm            — FWHM = 2.355·sigma in nm (blank if rejected)
        ellipticity                 — (σ_x − σ_y) / σ_xy (blank if rejected)
        snr                         — peak / std(outer-shell bg) (blank if rejected)
        """
        if self._bead_data is None:
            messagebox.showwarning("No data", "Run the estimation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export bead table as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        bd       = self._bead_data
        dx       = bd['dx']
        dz       = bd['dz']
        is_batch = 'volume_paths' in bd

        FWHM = 2.355 * 1000   # µm → nm conversion factor

        fieldnames = ['bead_id']
        if is_batch:
            fieldnames += ['volume_id', 'source_file']
        fieldnames += [
            'z_px', 'y_px', 'x_px',
            'z_um', 'y_um', 'x_um',
            'status',
            'sigma_z_um', 'sigma_y_um', 'sigma_x_um', 'sigma_xy_um',
            'fwhm_z_nm',  'fwhm_y_nm',  'fwhm_x_nm',  'fwhm_xy_nm',
            'ellipticity', 'snr',
        ]

        rows    = []
        bead_id = 0

        _empty = {
            'sigma_z_um': '', 'sigma_y_um': '', 'sigma_x_um': '', 'sigma_xy_um': '',
            'fwhm_z_nm':  '', 'fwhm_y_nm':  '', 'fwhm_x_nm':  '', 'fwhm_xy_nm':  '',
            'ellipticity': '', 'snr': '',
        }

        def _vol_fields(vid):
            if not is_batch:
                return {}
            src = os.path.basename(bd['volume_paths'][vid])
            return {'volume_id': int(vid), 'source_file': src}

        # Border-rejected
        border_vids = bd.get('border_volume_id', [])
        for j, (z, y, x) in enumerate(bd['border_px']):
            vid = int(border_vids[j]) if is_batch and j < len(border_vids) else 0
            rows.append({
                'bead_id': bead_id,
                **_vol_fields(vid),
                'z_px': z,  'y_px': y,  'x_px': x,
                'z_um': f"{z*dz:.4f}", 'y_um': f"{y*dx:.4f}", 'x_um': f"{x*dx:.4f}",
                'status': 'border_rejected',
                **_empty,
            })
            bead_id += 1

        # Quality-rejected
        rejected_vids = bd.get('rejected_volume_id', [])
        for j, (z, y, x) in enumerate(bd['rejected_px']):
            vid = int(rejected_vids[j]) if is_batch and j < len(rejected_vids) else 0
            rows.append({
                'bead_id': bead_id,
                **_vol_fields(vid),
                'z_px': z,  'y_px': y,  'x_px': x,
                'z_um': f"{z*dz:.4f}", 'y_um': f"{y*dx:.4f}", 'x_um': f"{x*dx:.4f}",
                'status': 'quality_rejected',
                **_empty,
            })
            bead_id += 1

        # Accepted beads (with sigma data)
        acc_px    = bd['accepted_px']
        sz        = bd['accepted_sigma_z']
        sy        = bd['accepted_sigma_y']
        sx        = bd['accepted_sigma_x']
        sxy       = bd['accepted_sigma_xy']
        ell       = bd['accepted_ellipticity']
        snr       = bd['accepted_snr']
        used      = bd['accepted_used']
        acc_vids  = bd.get('volume_id', [])

        for i in range(len(acc_px)):
            z, y, x = acc_px[i]
            vid = int(acc_vids[i]) if is_batch and i < len(acc_vids) else 0
            rows.append({
                'bead_id': bead_id,
                **_vol_fields(vid),
                'z_px': z,  'y_px': y,  'x_px': x,
                'z_um': f"{z*dz:.4f}", 'y_um': f"{y*dx:.4f}", 'x_um': f"{x*dx:.4f}",
                'status': 'used_in_psf' if used[i] else 'accepted',
                'sigma_z_um':  f"{sz[i]:.4f}",
                'sigma_y_um':  f"{sy[i]:.4f}",
                'sigma_x_um':  f"{sx[i]:.4f}",
                'sigma_xy_um': f"{sxy[i]:.4f}",
                'fwhm_z_nm':   f"{sz[i]  * FWHM:.1f}",
                'fwhm_y_nm':   f"{sy[i]  * FWHM:.1f}",
                'fwhm_x_nm':   f"{sx[i]  * FWHM:.1f}",
                'fwhm_xy_nm':  f"{sxy[i] * FWHM:.1f}",
                'ellipticity': f"{ell[i]:.4f}",
                'snr':         f"{snr[i]:.2f}",
            })
            bead_id += 1

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            messagebox.showinfo(
                "Export complete",
                f"Bead table saved to:\n{path}\n\n"
                f"{len(rows)} beads total  "
                f"({bd['n_used']} used in PSF, "
                f"{bd['n_accepted'] - bd['n_used']} accepted, "
                f"{bd['n_quality_rejected']} quality-rejected, "
                f"{bd['n_border']} border-rejected)",
            )
            self.status_var.set(f"CSV exported: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    # =========================================================================
    # Log helpers
    # =========================================================================

    def _log_append(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")
        self.log.update_idletasks()

    def _clear_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    # =========================================================================
    # Entry point
    # =========================================================================

    def run(self):
        self.root.mainloop()


# =============================================================================
# Stdout redirector (worker thread → log widget)
# =============================================================================

class _StdoutRedirector:
    def __init__(self, write_fn):
        self._write = write_fn

    def write(self, s):
        self._write(s)

    def flush(self):
        pass


# =============================================================================
# Public entry point
# =============================================================================

def launch_gui():
    app = PSFScopeGUI()
    app.run()


if __name__ == "__main__":
    launch_gui()
