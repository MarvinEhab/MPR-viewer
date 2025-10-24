#!/usr/bin/env python3
"""
MPV (Multi-Planar View) DICOM Viewer with DUAL AI Detection
- FIXED: Correct ImageOrientationPatient for exported DICOM files
- Organ Detection (TotalSegmentator)
- Orientation Detection (3D CNN)
- Export with orientation selection
- Oblique plane visualization
- Enhanced NIfTI organ viewer with OUTLINE/CONTOUR mode
Complete integrated version
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector, RadioButtons
from matplotlib.patches import Rectangle
import pydicom
from pydicom.dataset import Dataset, FileDataset
from tkinter import Tk, filedialog, messagebox
import tkinter as tk
import imageio
from datetime import datetime
from scipy import ndimage
import os
import threading

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# Import AI detectors
try:
    from ai_organ_detector import OrganDetector
    HAS_ORGAN_DETECTOR = True
    print("✓ Organ Detector loaded")
except ImportError:
    HAS_ORGAN_DETECTOR = False
    print("⚠ Organ Detector not available")

try:
    from ai_orientation_detector import OrientationDetector
    HAS_ORIENTATION_DETECTOR = True
    print("✓ Orientation Detector loaded")
except ImportError:
    HAS_ORIENTATION_DETECTOR = False
    print("⚠ Orientation Detector not available")

print("=== MPV Viewer with DUAL AI Detection ===")


class MPVViewer:
    def __init__(self, volume_data, metadata, organ_volume=None, dicom_files=None, dicom_path=None):
        """Initialize MPV viewer with DUAL AI integration"""
        self.volume = volume_data
        self.metadata = metadata
        self.organ_volume = organ_volume
        self.dicom_files = dicom_files
        self.dicom_path = dicom_path
        
        # AI Detection results
        self.ai_detected_organs = []
        self.ai_detected_orientation = {}
        self.ai_detection_running = False
        self.ai_organ_detector = None
        self.ai_orientation_detector = None
        
        # Dimensions
        self.nz, self.ny, self.nx = volume_data.shape
        print(f"Volume: {self.nx} x {self.ny} x {self.nz}")
        
        # Detect organ orientation if loaded
        self.organ_axis_mapping = None
        if organ_volume is not None:
            self.detect_organ_orientation()
        
        # Cursor
        self.cursor_x = self.nx // 2
        self.cursor_y = self.ny // 2
        self.cursor_z = self.nz // 2
        
        # Window/Level
        self.window_center = np.median(volume_data)
        self.window_width = np.percentile(volume_data, 95) - np.percentile(volume_data, 5)
        self.default_center = self.window_center
        self.default_width = self.window_width
        
        # Zoom
        self.zoom_level = 1.0
        
        # Pan offsets
        self.pan_offsets = {
            'axial': [0, 0],
            'sagittal': [0, 0],
            'coronal': [0, 0],
            '4th': [0, 0]
        }
        self.pan_active = None
        self.pan_start = None
        
        # Oblique parameters
        self.oblique_rotation_xy = 0
        self.oblique_rotation_z = 0
        self.oblique_center_x = self.nx // 2
        self.oblique_center_y = self.ny // 2
        self.oblique_center_z = self.nz // 2
        self.oblique_cache = None
        self.oblique_cache_params = None
        
        # View modes
        self.view_mode = 'oblique'
        self.show_oblique_in = 'fourth'
        self.organ_view_axis = 'axial'
        
        # ADDED: Organ slice tracking for independent scrolling
        self.organ_slice_idx = None  # Will be initialized when organ loads
        
        # ROI selection
        self.roi_active = False
        self.roi_coords = None
        self.roi_view = None
        self.roi_limit_navigation = False
        self.roi_drawing_mode = False
        
        # Cine mode
        self.cine_active = False
        self.cine_timer = None
        self.cine_speed = 100
        self.cine_direction = 1
        self.cine_axis = 'z'
        
        # Performance tracking
        self.update_in_progress = False
        
        # Create figure
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.canvas.manager.set_window_title('MPV Viewer - DUAL AI Enhanced (FIXED Export)')
        
        # Create main views
        gs = self.fig.add_gridspec(2, 4, left=0.05, right=0.95, top=0.95, bottom=0.35, 
                                    hspace=0.15, wspace=0.15)
        self.ax_axial = self.fig.add_subplot(gs[0, 0])
        self.ax_sagittal = self.fig.add_subplot(gs[0, 1])
        self.ax_coronal = self.fig.add_subplot(gs[0, 2])
        self.ax_4th = self.fig.add_subplot(gs[0, 3])
        
        # Info panels
        self.ax_info = self.fig.add_axes([0.05, 0.285, 0.9, 0.03])
        self.ax_info.axis('off')
        
        # DUAL AI Detection panel
        self.ax_ai_panel = self.fig.add_axes([0.05, 0.245, 0.9, 0.03])
        self.ax_ai_panel.axis('off')
        
        # Navigation sliders
        self.ax_slice_x_slider = self.fig.add_axes([0.15, 0.21, 0.7, 0.015])
        self.ax_slice_y_slider = self.fig.add_axes([0.15, 0.19, 0.7, 0.015])
        self.ax_slice_z_slider = self.fig.add_axes([0.15, 0.17, 0.7, 0.015])
        
        # Control sliders
        self.ax_window_slider = self.fig.add_axes([0.15, 0.13, 0.7, 0.015])
        self.ax_level_slider = self.fig.add_axes([0.15, 0.11, 0.7, 0.015])
        
        # Oblique controls
        self.ax_oblique_xy_slider = self.fig.add_axes([0.15, 0.07, 0.7, 0.015])
        self.ax_oblique_z_slider = self.fig.add_axes([0.15, 0.05, 0.7, 0.015])
        
        # Cine speed
        self.ax_cine_slider = self.fig.add_axes([0.15, 0.03, 0.7, 0.015])
        
        # Buttons
        button_width = 0.065
        button_height = 0.028
        button_y1 = 0.005
        
        self.ax_upload_dcm_btn = self.fig.add_axes([0.05, button_y1, button_width, button_height])
        self.ax_upload_nii_btn = self.fig.add_axes([0.12, button_y1, button_width, button_height])
        self.ax_export_btn = self.fig.add_axes([0.19, button_y1, button_width, button_height])
        self.ax_reset_btn = self.fig.add_axes([0.26, button_y1, button_width, button_height])
        self.ax_zoom_in_btn = self.fig.add_axes([0.33, button_y1, button_width, button_height])
        self.ax_zoom_out_btn = self.fig.add_axes([0.40, button_y1, button_width, button_height])
        self.ax_reset_pan_btn = self.fig.add_axes([0.47, button_y1, button_width, button_height])
        self.ax_roi_btn = self.fig.add_axes([0.55, button_y1, button_width, button_height])
        self.ax_roi_limit_btn = self.fig.add_axes([0.62, button_y1, button_width, button_height])
        self.ax_clear_roi_btn = self.fig.add_axes([0.69, button_y1, button_width, button_height])
        self.ax_cine_btn = self.fig.add_axes([0.76, button_y1, button_width, button_height])
        self.ax_cine_axis_btn = self.fig.add_axes([0.83, button_y1, button_width, button_height])
        
        if organ_volume is not None:
            self.ax_mode_btn = self.fig.add_axes([0.90, button_y1, 0.09, button_height])
            self.ax_organ_axis = self.fig.add_axes([0.90, 0.04, 0.09, 0.08])
            self.ax_organ_axis.set_visible(False)
        
        # Set titles
        self.ax_axial.set_title('Axial (Z-axis)', fontsize=11, fontweight='bold')
        self.ax_sagittal.set_title('Sagittal (X-axis)', fontsize=11, fontweight='bold')
        self.ax_coronal.set_title('Coronal (Y-axis)', fontsize=11, fontweight='bold')
        self.ax_4th.set_title('Oblique View', fontsize=11, fontweight='bold')
        
        # Image displays
        self.img_axial = None
        self.img_sagittal = None
        self.img_coronal = None
        self.img_4th = None
        
        # Crosshairs
        self.crosshairs = {
            'axial': {'h': None, 'v': None},
            'sagittal': {'h': None, 'v': None},
            'coronal': {'h': None, 'v': None}
        }
        
        # Oblique indicators
        self.oblique_indicators = {
            'axial': None,
            'sagittal': None,
            'coronal': None
        }
        
        # ROI rectangles
        self.roi_rectangles = {}
        
        self.setup_widgets()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Initial display
        self.update_all_views()
        self.update_info_panel()
        self.update_ai_panel("Initializing...")
        
        # Start DUAL AI detection if path available
        if (HAS_ORGAN_DETECTOR or HAS_ORIENTATION_DETECTOR) and self.dicom_path:
            self.start_ai_detection()
               
    def start_ai_detection(self):
        """Start BOTH organ and orientation detection in background thread"""
        if self.ai_detection_running:
            return
        
        self.ai_detection_running = True
        self.update_ai_panel("🤖 DUAL AI Detection running... (Organ + Orientation)")
        
        def run_detection():
            try:
                organ_results = None
                orientation_results = None
                
                # 1. Organ Detection
                if HAS_ORGAN_DETECTOR:
                    print("\n🔬 Starting Organ Detection...")
                    if self.ai_organ_detector is None:
                        self.ai_organ_detector = OrganDetector(fast_mode=True, use_gpu=True)
                    organ_results = self.ai_organ_detector.detect_organs(self.dicom_path)
                
                # 2. Orientation Detection
                if HAS_ORIENTATION_DETECTOR:
                    print("\n🧭 Starting Orientation Detection...")
                    try:
                        if self.ai_orientation_detector is None:
                            self.ai_orientation_detector = OrientationDetector(
                                model_path='best_orientation_classifier.pth'
                            )
                        orientation_results = self.ai_orientation_detector.detect_orientation(self.dicom_path)
                    except Exception as e:
                        print(f"⚠ Orientation detection failed: {e}")
                        orientation_results = {'orientation': 'unknown', 'confidence': 0.0}
                
                # Store results
                if organ_results and 'top_organs' in organ_results and organ_results['top_organs']:
                    self.ai_detected_organs = organ_results['top_organs'][:3]
                else:
                    self.ai_detected_organs = []
                
                if orientation_results:
                    self.ai_detected_orientation = orientation_results
                else:
                    self.ai_detected_orientation = {'orientation': 'unknown', 'confidence': 0.0}
                
                # Update UI
                self.fig.canvas.draw_idle()
                
            except Exception as e:
                print(f"❌ AI Detection Error: {e}")
                import traceback
                traceback.print_exc()
                self.ai_detected_organs = [('error', 0)]
                self.ai_detected_orientation = {'orientation': 'error', 'confidence': 0.0}
            finally:
                self.ai_detection_running = False
                self.update_ai_panel()
                self.fig.canvas.draw_idle()
        
        # Run in background thread
        detection_thread = threading.Thread(target=run_detection, daemon=True)
        detection_thread.start()
    
    def update_ai_panel(self, message=None):
        """Update DUAL AI detection info panel"""
        self.ax_ai_panel.clear()
        self.ax_ai_panel.axis('off')
        
        if message:
            self.ax_ai_panel.text(0.5, 0.5, message, ha='center', va='center', 
                                 fontsize=9, bbox=dict(boxstyle='round', 
                                 facecolor='lightyellow', alpha=0.8))
        elif hasattr(self, 'ai_detected_organs') and hasattr(self, 'ai_detected_orientation'):
            if self.ai_detected_organs and self.ai_detected_organs[0][0] == 'error':
                ai_text = "⚠ AI Detection failed - check console"
                color = 'lightcoral'
            elif self.ai_detected_organs or self.ai_detected_orientation:
                organ_text = ""
                if self.ai_detected_organs:
                    organ_names = [name.replace('_', ' ').title() for name, _ in self.ai_detected_organs]
                    organ_text = f"🔬 Organs: 1️⃣ {organ_names[0]}"
                    if len(organ_names) > 1:
                        organ_text += f" | 2️⃣ {organ_names[1]}"
                    if len(organ_names) > 2:
                        organ_text += f" | 3️⃣ {organ_names[2]}"
                
                orientation = self.ai_detected_orientation
                orient_name = orientation.get('orientation', 'unknown').upper()
                orient_conf = orientation.get('confidence', 0.0)
                orient_text = f"🧭 {orient_name} ({orient_conf*100:.0f}%)"
                
                ai_text = f"🤖 AI: {orient_text}"
                if organ_text:
                    ai_text += f" | {organ_text}"
                
                color = 'lightgreen'
            else:
                ai_text = "🤖 DUAL AI Detection: Processing..."
                color = 'lightyellow'
            
            self.ax_ai_panel.text(0.5, 0.5, ai_text, ha='center', va='center', 
                                 fontsize=9, bbox=dict(boxstyle='round', 
                                 facecolor=color, alpha=0.8))
        else:
            self.ax_ai_panel.text(0.5, 0.5, "ℹ️ DUAL AI Detection: Not available", 
                                 ha='center', va='center', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def detect_organ_orientation(self):
        """Detect organ orientation"""
        organ_shape = self.organ_volume.shape
        main_shape = (self.nz, self.ny, self.nx)
        
        print(f"Detecting organ orientation...")
        print(f"  Main: {main_shape}, Organ: {organ_shape}")
        
        self.organ_axis_mapping = {
            'axial': None,
            'sagittal': None,
            'coronal': None
        }
        
        if organ_shape == main_shape:
            self.organ_axis_mapping = {
                'axial': (0, False),
                'sagittal': (2, False),
                'coronal': (1, False)
            }
        elif organ_shape == (main_shape[0], main_shape[2], main_shape[1]):
            self.organ_axis_mapping = {
                'axial': (0, True),
                'sagittal': (1, False),
                'coronal': (2, False)
            }
        elif organ_shape == (main_shape[2], main_shape[1], main_shape[0]):
            self.organ_axis_mapping = {
                'axial': (2, False),
                'sagittal': (0, False),
                'coronal': (1, False)
            }
        elif organ_shape == (main_shape[1], main_shape[0], main_shape[2]):
            self.organ_axis_mapping = {
                'axial': (1, False),
                'sagittal': (2, False),
                'coronal': (0, False)
            }
        else:
            self.organ_axis_mapping = {
                'axial': (0, False),
                'sagittal': (2, False),
                'coronal': (1, False)
            }
    
    def setup_widgets(self):
        """Setup all sliders and buttons"""
        self.slice_x_slider = Slider(
            self.ax_slice_x_slider, 'X Position',
            0, max(0, self.nx - 1), valinit=self.cursor_x, valstep=1, color='cyan'
        )
        self.slice_y_slider = Slider(
            self.ax_slice_y_slider, 'Y Position',
            0, max(0, self.ny - 1), valinit=self.cursor_y, valstep=1, color='magenta'
        )
        self.slice_z_slider = Slider(
            self.ax_slice_z_slider, 'Z Position',
            0, max(0, self.nz - 1), valinit=self.cursor_z, valstep=1, color='yellow'
        )
        
        self.slice_x_slider.on_changed(self.update_from_slider)
        self.slice_y_slider.on_changed(self.update_from_slider)
        self.slice_z_slider.on_changed(self.update_from_slider)
        
        self.window_slider = Slider(
            self.ax_window_slider, 'Window Width',
            self.window_width * 0.1, self.window_width * 3,
            valinit=self.window_width, valstep=10, color='blue'
        )
        self.level_slider = Slider(
            self.ax_level_slider, 'Window Center',
            np.min(self.volume), np.max(self.volume),
            valinit=self.window_center, valstep=5, color='green'
        )
        
        self.oblique_xy_slider = Slider(
            self.ax_oblique_xy_slider, 'Oblique Rotation',
            -180, 180, valinit=0, valstep=5, color='red'
        )
        self.oblique_z_slider = Slider(
            self.ax_oblique_z_slider, 'Oblique Elevation',
            -90, 90, valinit=0, valstep=5, color='orange'
        )
        
        self.cine_slider = Slider(
            self.ax_cine_slider, 'Cine Speed (fps)',
            1, 30, valinit=10, valstep=1, color='purple'
        )
        
        self.window_slider.on_changed(self.update_window_level)
        self.level_slider.on_changed(self.update_window_level)
        self.oblique_xy_slider.on_changed(self.update_oblique_rotation)
        self.oblique_z_slider.on_changed(self.update_oblique_rotation)
        self.cine_slider.on_changed(self.update_cine_speed)
        
        self.upload_dcm_btn = Button(self.ax_upload_dcm_btn, 'Load\nDICOM', color='lightblue', hovercolor='skyblue')
        self.upload_dcm_btn.on_clicked(self.load_dicom_dialog)
        
        self.upload_nii_btn = Button(self.ax_upload_nii_btn, 'Load\nNIfTI', color='lightblue', hovercolor='skyblue')
        self.upload_nii_btn.on_clicked(self.load_nifti_dialog)
        
        self.export_btn = Button(self.ax_export_btn, 'Export\nDICOM', color='lightgreen', hovercolor='lightgreen')
        self.export_btn.on_clicked(self.export_dicom)
        
        self.reset_btn = Button(self.ax_reset_btn, 'Reset\nW/L', color='lightgray', hovercolor='silver')
        self.reset_btn.on_clicked(self.reset_window_level)
        
        self.zoom_in_btn = Button(self.ax_zoom_in_btn, 'Zoom\n+', color='lightgray', hovercolor='silver')
        self.zoom_in_btn.on_clicked(lambda event: self.adjust_zoom(1.2))
        
        self.zoom_out_btn = Button(self.ax_zoom_out_btn, 'Zoom\n-', color='lightgray', hovercolor='silver')
        self.zoom_out_btn.on_clicked(lambda event: self.adjust_zoom(0.8))
        
        self.reset_pan_btn = Button(self.ax_reset_pan_btn, 'Reset\nPan', color='lightgray', hovercolor='silver')
        self.reset_pan_btn.on_clicked(self.reset_pan)
        
        self.roi_btn = Button(self.ax_roi_btn, 'Define\nROI', color='yellow', hovercolor='gold')
        self.roi_btn.on_clicked(self.activate_roi_selection)
        
        self.roi_limit_btn = Button(self.ax_roi_limit_btn, 'Limit:\nOFF', color='orange', hovercolor='darkorange')
        self.roi_limit_btn.on_clicked(self.toggle_roi_limit)
        
        self.clear_roi_btn = Button(self.ax_clear_roi_btn, 'Clear\nROI', color='lightyellow', hovercolor='khaki')
        self.clear_roi_btn.on_clicked(self.clear_roi)
        
        self.cine_btn = Button(self.ax_cine_btn, 'Start\nCine', color='lightcoral', hovercolor='salmon')
        self.cine_btn.on_clicked(self.toggle_cine)
        
        self.cine_axis_btn = Button(self.ax_cine_axis_btn, f'Cine:\nAxial', color='lightsalmon', hovercolor='coral')
        self.cine_axis_btn.on_clicked(self.cycle_cine_axis)
        
        if self.organ_volume is not None:
            self.mode_btn = Button(self.ax_mode_btn, 'Mode:\nOblique', color='lightgray', hovercolor='silver')
            self.mode_btn.on_clicked(self.toggle_view_mode)
            
            self.organ_axis_radio = RadioButtons(self.ax_organ_axis, ('Axial', 'Sagittal', 'Coronal'))
            self.organ_axis_radio.on_clicked(self.set_organ_axis)
    
    def set_organ_axis(self, label):
        """Set organ axis and reset to middle slice"""
        self.organ_view_axis = label.lower()
        
        # Reset to middle slice when changing axis
        if self.organ_volume is not None:
            organ_shape = self.organ_volume.shape
            if self.organ_view_axis == 'axial':
                self.organ_slice_idx = organ_shape[0] // 2
            elif self.organ_view_axis == 'sagittal':
                self.organ_slice_idx = organ_shape[2] // 2 if len(organ_shape) > 2 else organ_shape[0] // 2
            elif self.organ_view_axis == 'coronal':
                self.organ_slice_idx = organ_shape[1] // 2 if len(organ_shape) > 1 else organ_shape[0] // 2
        
        self.update_all_views()
    
    def reset_pan(self, event):
        self.pan_offsets = {'axial': [0, 0], 'sagittal': [0, 0], 'coronal': [0, 0], '4th': [0, 0]}
        print("✓ Pan reset")
        self.update_all_views()
    
    def get_roi_limits(self):
        if self.roi_limit_navigation and self.roi_coords:
            return self.roi_coords
        else:
            return {'x': (0, self.nx - 1), 'y': (0, self.ny - 1), 'z': (0, self.nz - 1)}
    
    def toggle_roi_limit(self, event):
        if not self.roi_coords:
            print("⚠ Define ROI first!")
            return
        
        self.roi_limit_navigation = not self.roi_limit_navigation
        
        if self.roi_limit_navigation:
            self.roi_limit_btn.label.set_text('Limit:\nON')
            self.roi_limit_btn.color = 'limegreen'
            
            limits = self.get_roi_limits()
            self.cursor_x = np.clip(self.cursor_x, limits['x'][0], limits['x'][1])
            self.cursor_y = np.clip(self.cursor_y, limits['y'][0], limits['y'][1])
            self.cursor_z = np.clip(self.cursor_z, limits['z'][0], limits['z'][1])
            
            self.update_in_progress = True
            self.slice_x_slider.set_val(self.cursor_x)
            self.slice_y_slider.set_val(self.cursor_y)
            self.slice_z_slider.set_val(self.cursor_z)
            self.update_in_progress = False
        else:
            self.roi_limit_btn.label.set_text('Limit:\nOFF')
            self.roi_limit_btn.color = 'orange'
        
        self.update_all_views()
        self.update_info_panel()
    
    def update_from_slider(self, val):
        if self.update_in_progress:
            return
        
        limits = self.get_roi_limits()
        
        self.cursor_x = int(np.clip(self.slice_x_slider.val, limits['x'][0], limits['x'][1]))
        self.cursor_y = int(np.clip(self.slice_y_slider.val, limits['y'][0], limits['y'][1]))
        self.cursor_z = int(np.clip(self.slice_z_slider.val, limits['z'][0], limits['z'][1]))
        
        self.update_all_views()
        self.update_info_panel()
    
    def apply_window_level(self, image):
        """Apply window/level"""
        vmin = self.window_center - self.window_width / 2
        vmax = self.window_center + self.window_width / 2
        windowed = np.clip(image, vmin, vmax)
        windowed = (windowed - vmin) / (vmax - vmin + 1e-6)
        return windowed
    
    def update_window_level(self, val):
        """Update window/level"""
        self.window_width = self.window_slider.val
        self.window_center = self.level_slider.val
        self.update_all_views()
    
    def update_oblique_rotation(self, val):
        """Update oblique rotation"""
        self.oblique_rotation_xy = self.oblique_xy_slider.val
        self.oblique_rotation_z = self.oblique_z_slider.val
        self.oblique_cache = None
        self.update_all_views()
    
    def update_cine_speed(self, val):
        """Update cine speed"""
        fps = self.cine_slider.val
        self.cine_speed = int(1000 / fps)
    
    def reset_window_level(self, event):
        """Reset window/level"""
        self.window_width = self.default_width
        self.window_center = self.default_center
        self.window_slider.set_val(self.window_width)
        self.level_slider.set_val(self.window_center)
        self.update_all_views()
    
    def adjust_zoom(self, factor):
        """Adjust zoom"""
        self.zoom_level = np.clip(self.zoom_level * factor, 0.5, 10.0)
        print(f"Zoom: {self.zoom_level:.1f}x")
        self.update_all_views()
    
    def toggle_view_mode(self, event):
        """Toggle view mode"""
        if self.organ_volume is None:
            return
        
        self.view_mode = 'organ' if self.view_mode == 'oblique' else 'oblique'
        
        if self.view_mode == 'organ':
            self.mode_btn.label.set_text('Mode:\nOrgan')
            self.ax_organ_axis.set_visible(True)
        else:
            self.mode_btn.label.set_text('Mode:\nOblique')
            self.ax_organ_axis.set_visible(False)
        
        self.fig.canvas.draw_idle()
        self.update_all_views()
    
    def cycle_cine_axis(self, event):
        """Cycle cine axis"""
        axes = ['z', 'x', 'y']
        current_idx = axes.index(self.cine_axis)
        self.cine_axis = axes[(current_idx + 1) % len(axes)]
        
        axis_names = {'z': 'Axial', 'x': 'Sagittal', 'y': 'Coronal'}
        self.cine_axis_btn.label.set_text(f'Cine:\n{axis_names[self.cine_axis]}')
           
    def get_oblique_slice_optimized(self):
        """Get oblique slice"""
        cache_params = (self.oblique_rotation_xy, self.oblique_rotation_z, 
                       self.oblique_center_x, self.oblique_center_y, self.oblique_center_z)
        
        if self.oblique_cache is not None and self.oblique_cache_params == cache_params:
            return self.oblique_cache
        
        angle_xy = np.radians(self.oblique_rotation_xy)
        angle_z = np.radians(self.oblique_rotation_z)
        
        normal = np.array([
            np.sin(angle_xy) * np.cos(angle_z),
            np.cos(angle_xy) * np.cos(angle_z),
            np.sin(angle_z)
        ])
        
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([0, 1, 0])
        
        right = np.cross(normal, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, normal)
        up = up / (np.linalg.norm(up) + 1e-8)
        
        slice_width = min(512, max(self.nx, self.ny))
        slice_height = min(512, max(self.nz, max(self.nx, self.ny)))
        
        center = np.array([self.oblique_center_x, self.oblique_center_y, self.oblique_center_z])
        
        u_coords = np.linspace(-max(self.nx, self.ny) / 2, max(self.nx, self.ny) / 2, slice_width)
        v_coords = np.linspace(-max(self.nz, max(self.nx, self.ny)) / 2, 
                               max(self.nz, max(self.nx, self.ny)) / 2, slice_height)
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)
        
        x_coords = center[0] + u_grid * right[0] + v_grid * up[0]
        y_coords = center[1] + u_grid * right[1] + v_grid * up[1]
        z_coords = center[2] + u_grid * right[2] + v_grid * up[2]
        
        coords = np.array([z_coords, y_coords, x_coords])
        oblique_slice = ndimage.map_coordinates(self.volume, coords, order=1, 
                                                 mode='constant', cval=0)
        
        self.oblique_cache = oblique_slice
        self.oblique_cache_params = cache_params
        
        return oblique_slice
    
    def get_organ_slice(self):
        """Get organ slice with OUTLINE/CONTOUR mode - multi-colored outlines"""
        if self.organ_volume is None or self.organ_axis_mapping is None:
            return np.zeros((self.ny, self.nx))
        
        organ_shape = self.organ_volume.shape
        
        # Initialize organ slice index if not set (center slice)
        if self.organ_slice_idx is None:
            if self.organ_view_axis == 'axial':
                self.organ_slice_idx = organ_shape[0] // 2
            elif self.organ_view_axis == 'sagittal':
                self.organ_slice_idx = organ_shape[2] // 2 if len(organ_shape) > 2 else organ_shape[0] // 2
            elif self.organ_view_axis == 'coronal':
                self.organ_slice_idx = organ_shape[1] // 2 if len(organ_shape) > 1 else organ_shape[0] // 2
        
        try:
            # Get the slice based on current axis and index
            if self.organ_view_axis == 'axial':
                axis_info = self.organ_axis_mapping['axial']
                if axis_info is None:
                    slice_idx = min(self.organ_slice_idx, organ_shape[0]-1)
                    slice_data = self.organ_volume[slice_idx, :, :]
                else:
                    axis, transpose = axis_info
                    
                    if axis == 0:
                        slice_idx = min(self.organ_slice_idx, organ_shape[0]-1)
                        slice_data = self.organ_volume[slice_idx, :, :]
                    elif axis == 1:
                        slice_idx = min(self.organ_slice_idx, organ_shape[1]-1)
                        slice_data = self.organ_volume[:, slice_idx, :]
                    elif axis == 2:
                        slice_idx = min(self.organ_slice_idx, organ_shape[2]-1)
                        slice_data = self.organ_volume[:, :, slice_idx]
                    
                    if transpose:
                        slice_data = slice_data.T
            
            elif self.organ_view_axis == 'sagittal':
                axis_info = self.organ_axis_mapping['sagittal']
                if axis_info is None:
                    slice_idx = min(self.organ_slice_idx, organ_shape[2]-1)
                    slice_data = self.organ_volume[:, :, slice_idx]
                else:
                    axis, transpose = axis_info
                    
                    if axis == 0:
                        slice_idx = min(self.organ_slice_idx, organ_shape[0]-1)
                        slice_data = self.organ_volume[slice_idx, :, :]
                    elif axis == 1:
                        slice_idx = min(self.organ_slice_idx, organ_shape[1]-1)
                        slice_data = self.organ_volume[:, slice_idx, :]
                    elif axis == 2:
                        slice_idx = min(self.organ_slice_idx, organ_shape[2]-1)
                        slice_data = self.organ_volume[:, :, slice_idx]
                    
                    if transpose:
                        slice_data = slice_data.T
            
            elif self.organ_view_axis == 'coronal':
                axis_info = self.organ_axis_mapping['coronal']
                if axis_info is None:
                    slice_idx = min(self.organ_slice_idx, organ_shape[1]-1)
                    slice_data = self.organ_volume[:, slice_idx, :]
                else:
                    axis, transpose = axis_info
                    
                    if axis == 0:
                        slice_idx = min(self.organ_slice_idx, organ_shape[0]-1)
                        slice_data = self.organ_volume[slice_idx, :, :]
                    elif axis == 1:
                        slice_idx = min(self.organ_slice_idx, organ_shape[1]-1)
                        slice_data = self.organ_volume[:, slice_idx, :]
                    elif axis == 2:
                        slice_idx = min(self.organ_slice_idx, organ_shape[2]-1)
                        slice_data = self.organ_volume[:, :, slice_idx]
                    
                    if transpose:
                        slice_data = slice_data.T
            
            # ENHANCED: Multi-organ contour detection (OUTLINE MODE)
            # Get unique organ labels
            unique_labels = np.unique(slice_data)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            
            # Create output image for all contours
            contour_image = np.zeros_like(slice_data, dtype=float)
            
            # Extract contours for each organ
            for label in unique_labels:
                # Create binary mask for this organ
                organ_mask = (slice_data == label).astype(np.uint8)
                
                # Detect edges using Sobel filter
                edges_x = ndimage.sobel(organ_mask, axis=0)
                edges_y = ndimage.sobel(organ_mask, axis=1)
                edges = np.hypot(edges_x, edges_y)
                
                # Normalize
                if edges.max() > 0:
                    edges = edges / edges.max()
                
                # Dilate edges slightly to make them more visible
                edges = ndimage.binary_dilation(edges > 0.1, iterations=1).astype(float)
                
                # Add to contour image (scaled by label value for different colors)
                contour_image = np.maximum(contour_image, edges * label)
            
            # Normalize final output
            if contour_image.max() > 0:
                contour_image = contour_image / contour_image.max()
            
            return contour_image
        
        except Exception as e:
            print(f"ERROR in get_organ_slice: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((max(self.ny, self.nx), max(self.ny, self.nx)))
    
    def get_view_slice(self, view_name):
        """Get view slice"""
        if view_name == 'axial':
            if self.show_oblique_in == 'axial' and self.view_mode == 'oblique':
                return self.get_oblique_slice_optimized()
            return self.volume[self.cursor_z, :, :]
        
        elif view_name == 'sagittal':
            if self.show_oblique_in == 'sagittal' and self.view_mode == 'oblique':
                return self.get_oblique_slice_optimized()
            return self.volume[:, :, self.cursor_x]
        
        elif view_name == 'coronal':
            if self.show_oblique_in == 'coronal' and self.view_mode == 'oblique':
                return self.get_oblique_slice_optimized()
            return self.volume[:, self.cursor_y, :]
        
        elif view_name == '4th':
            if self.show_oblique_in == 'fourth':
                if self.view_mode == 'oblique':
                    return self.get_oblique_slice_optimized()
                elif self.view_mode == 'organ' and self.organ_volume is not None:
                    return self.get_organ_slice()
            return self.volume[self.cursor_z, :, :]
        
        return None
    
    def update_all_views(self):
        """Update all views with ENHANCED organ outline display"""
        if self.update_in_progress:
            return
        
        self.update_in_progress = True
        
        try:
            self.cursor_x = np.clip(self.cursor_x, 0, self.nx - 1)
            self.cursor_y = np.clip(self.cursor_y, 0, self.ny - 1)
            self.cursor_z = np.clip(self.cursor_z, 0, self.nz - 1)
            
            axial_slice = self.get_view_slice('axial')
            sagittal_slice = self.get_view_slice('sagittal')
            coronal_slice = self.get_view_slice('coronal')
            fourth_slice = self.get_view_slice('4th')
            
            axial_w = self.apply_window_level(axial_slice)
            sagittal_w = self.apply_window_level(sagittal_slice)
            coronal_w = self.apply_window_level(coronal_slice)
            
            # ENHANCED: Special handling for organ outline view (don't apply window/level)
            if self.view_mode == 'organ' and self.show_oblique_in == 'fourth':
                fourth_w = fourth_slice  # Already normalized by get_organ_slice
            else:
                fourth_w = self.apply_window_level(fourth_slice)
            
            if self.img_axial is None:
                self.img_axial = self.ax_axial.imshow(axial_w, cmap='gray', aspect='equal', 
                                                        vmin=0, vmax=1, origin='lower', 
                                                        interpolation='bilinear')
                self.ax_axial.set_xlabel('X', fontsize=10)
                self.ax_axial.set_ylabel('Y', fontsize=10)
            else:
                self.img_axial.set_data(axial_w)
                self.img_axial.set_extent([0, axial_w.shape[1], 0, axial_w.shape[0]])
            
            if self.img_sagittal is None:
                self.img_sagittal = self.ax_sagittal.imshow(sagittal_w, cmap='gray', aspect='equal',
                                                             vmin=0, vmax=1, origin='lower',
                                                             interpolation='bilinear')
                self.ax_sagittal.set_xlabel('Y', fontsize=10)
                self.ax_sagittal.set_ylabel('Z', fontsize=10)
            else:
                self.img_sagittal.set_data(sagittal_w)
                self.img_sagittal.set_extent([0, sagittal_w.shape[1], 0, sagittal_w.shape[0]])
            
            if self.img_coronal is None:
                self.img_coronal = self.ax_coronal.imshow(coronal_w, cmap='gray', aspect='equal',
                                                           vmin=0, vmax=1, origin='lower',
                                                           interpolation='bilinear')
                self.ax_coronal.set_xlabel('X', fontsize=10)
                self.ax_coronal.set_ylabel('Z', fontsize=10)
            else:
                self.img_coronal.set_data(coronal_w)
                self.img_coronal.set_extent([0, coronal_w.shape[1], 0, coronal_w.shape[0]])
            
            # ENHANCED: Better colormap for organ outlines
            if self.view_mode == 'organ':
                cmap = 'hot'  # Black background, bright colorful edges
                vmin_val = 0.0
                vmax_val = 1.0
            else:
                cmap = 'gray'
                vmin_val = 0
                vmax_val = 1
            
            if self.img_4th is None:
                self.img_4th = self.ax_4th.imshow(fourth_w, cmap=cmap, aspect='equal',
                                                   vmin=vmin_val, vmax=vmax_val, origin='lower',
                                                   interpolation='bilinear')
            else:
                self.img_4th.set_data(fourth_w)
                self.img_4th.set_extent([0, fourth_w.shape[1], 0, fourth_w.shape[0]])
                self.img_4th.set_cmap(cmap)
                self.img_4th.set_clim(vmin=vmin_val, vmax=vmax_val)
            
            self.update_crosshairs()
            self.draw_oblique_indicators()
            self.draw_roi()
            
            roi_status = " [ROI LIMITED]" if self.roi_limit_navigation else ""
            
            self.ax_axial.set_title(
                f'Axial (Z={self.cursor_z}/{self.nz-1}){roi_status}',
                fontsize=10, fontweight='bold'
            )
            self.ax_sagittal.set_title(
                f'Sagittal (X={self.cursor_x}/{self.nx-1}){roi_status}',
                fontsize=10, fontweight='bold'
            )
            self.ax_coronal.set_title(
                f'Coronal (Y={self.cursor_y}/{self.ny-1}){roi_status}',
                fontsize=10, fontweight='bold'
            )
            
            # ENHANCED: Show slice number for organ outline view
            if self.show_oblique_in == 'fourth':
                if self.view_mode == 'organ':
                    view_descriptions = {'axial': 'Top View', 'sagittal': 'Side View', 'coronal': 'Front View'}
                    
                    # Get max slice for current axis
                    organ_shape = self.organ_volume.shape
                    if self.organ_view_axis == 'axial':
                        max_slice = organ_shape[0] - 1
                    elif self.organ_view_axis == 'sagittal':
                        max_slice = organ_shape[2] - 1 if len(organ_shape) > 2 else organ_shape[0] - 1
                    elif self.organ_view_axis == 'coronal':
                        max_slice = organ_shape[1] - 1 if len(organ_shape) > 1 else organ_shape[0] - 1
                    
                    self.ax_4th.set_title(
                        f'Organ Outlines - {self.organ_view_axis.capitalize()} - Slice {self.organ_slice_idx}/{max_slice}', 
                        fontsize=10, fontweight='bold'
                    )
                else:
                    self.ax_4th.set_title(
                        f'Oblique ({self.oblique_rotation_xy:.0f}°, {self.oblique_rotation_z:.0f}°)', 
                        fontsize=10, fontweight='bold'
                    )
            else:
                self.ax_4th.set_title(f'Axial (Z={self.cursor_z}/{self.nz-1})', fontsize=10, fontweight='bold')
            
            views_data = [
                (self.ax_axial, axial_w.shape, 'axial', self.cursor_x, self.cursor_y),
                (self.ax_sagittal, sagittal_w.shape, 'sagittal', self.cursor_y, self.cursor_z),
                (self.ax_coronal, coronal_w.shape, 'coronal', self.cursor_x, self.cursor_z),
                (self.ax_4th, fourth_w.shape, '4th', fourth_w.shape[1]//2, fourth_w.shape[0]//2)
            ]
            
            for ax, shape, view_name, center_x, center_y in views_data:
                h, w = shape
                zoom_w = w / self.zoom_level / 2
                zoom_h = h / self.zoom_level / 2
                pan_x, pan_y = self.pan_offsets[view_name]
                ax.set_xlim([center_x - zoom_w + pan_x, center_x + zoom_w + pan_x])
                ax.set_ylim([center_y - zoom_h + pan_y, center_y + zoom_h + pan_y])
            
            self.fig.canvas.draw_idle()
        
        finally:
            self.update_in_progress = False
               
    def draw_oblique_indicators(self):
        """Draw oblique plane indicators as yellow lines showing intersection"""
        # Clear old indicators
        for key in self.oblique_indicators:
            if self.oblique_indicators[key] is not None:
                if isinstance(self.oblique_indicators[key], list):
                    for item in self.oblique_indicators[key]:
                        try:
                            item.remove()
                        except:
                            pass
                else:
                    try:
                        self.oblique_indicators[key].remove()
                    except:
                        pass
            self.oblique_indicators[key] = None
        
        if self.view_mode != 'oblique':
            return
        
        # Calculate oblique plane normal and basis vectors
        angle_xy = np.radians(self.oblique_rotation_xy)
        angle_z = np.radians(self.oblique_rotation_z)
        
        # Normal vector to oblique plane
        normal = np.array([
            np.sin(angle_xy) * np.cos(angle_z),
            np.cos(angle_xy) * np.cos(angle_z),
            np.sin(angle_z)
        ])
        
        # Plane center point
        center = np.array([self.oblique_center_x, self.oblique_center_y, self.oblique_center_z])
        
        # === AXIAL VIEW (Z-slice) ===
        if self.show_oblique_in != 'axial':
            z_current = self.cursor_z
            
            if abs(normal[2]) > 0.001:
                t_offset = normal[2] * (z_current - center[2])
                
                if abs(normal[0]) > abs(normal[1]):
                    y_points = np.array([0, self.ny - 1])
                    x_points = -(normal[1] * (y_points - center[1]) + t_offset) / normal[0] + center[0]
                else:
                    x_points = np.array([0, self.nx - 1])
                    y_points = -(normal[0] * (x_points - center[0]) + t_offset) / normal[1] + center[1]
                
                line = self.ax_axial.plot(x_points, y_points, 'yellow', linewidth=2, 
                                          linestyle='--', alpha=0.8)[0]
                
                point = self.ax_axial.plot(self.oblique_center_x, self.oblique_center_y, 
                                           'ro', markersize=8, markeredgewidth=2, 
                                           markerfacecolor='yellow')[0]
                
                self.oblique_indicators['axial'] = [line, point]
        
        # === SAGITTAL VIEW (X-slice) ===
        if self.show_oblique_in != 'sagittal':
            x_current = self.cursor_x
            
            if abs(normal[0]) > 0.001:
                t_offset = normal[0] * (x_current - center[0])
                
                if abs(normal[1]) > abs(normal[2]):
                    z_points = np.array([0, self.nz - 1])
                    y_points = -(normal[2] * (z_points - center[2]) + t_offset) / normal[1] + center[1]
                else:
                    y_points = np.array([0, self.ny - 1])
                    z_points = -(normal[1] * (y_points - center[1]) + t_offset) / normal[2] + center[2]
                
                line = self.ax_sagittal.plot(y_points, z_points, 'yellow', linewidth=2,
                                             linestyle='--', alpha=0.8)[0]
                
                point = self.ax_sagittal.plot(self.oblique_center_y, self.oblique_center_z,
                                              'ro', markersize=8, markeredgewidth=2,
                                              markerfacecolor='yellow')[0]
                
                self.oblique_indicators['sagittal'] = [line, point]
        
        # === CORONAL VIEW (Y-slice) ===
        if self.show_oblique_in != 'coronal':
            y_current = self.cursor_y
            
            if abs(normal[1]) > 0.001:
                t_offset = normal[1] * (y_current - center[1])
                
                if abs(normal[0]) > abs(normal[2]):
                    z_points = np.array([0, self.nz - 1])
                    x_points = -(normal[2] * (z_points - center[2]) + t_offset) / normal[0] + center[0]
                else:
                    x_points = np.array([0, self.nx - 1])
                    z_points = -(normal[0] * (x_points - center[0]) + t_offset) / normal[2] + center[2]
                
                line = self.ax_coronal.plot(x_points, z_points, 'yellow', linewidth=2,
                                            linestyle='--', alpha=0.8)[0]
                
                point = self.ax_coronal.plot(self.oblique_center_x, self.oblique_center_z,
                                             'ro', markersize=8, markeredgewidth=2,
                                             markerfacecolor='yellow')[0]
                
                self.oblique_indicators['coronal'] = [line, point]
    
    def update_crosshairs(self):
        """Update crosshairs"""
        for view in self.crosshairs.values():
            if view['h'] is not None:
                try:
                    view['h'].remove()
                except:
                    pass
            if view['v'] is not None:
                try:
                    view['v'].remove()
                except:
                    pass
        
        self.crosshairs['axial']['h'] = self.ax_axial.axhline(
            y=self.cursor_y, color='lime', linewidth=1, alpha=0.7, linestyle='--')
        self.crosshairs['axial']['v'] = self.ax_axial.axvline(
            x=self.cursor_x, color='lime', linewidth=1, alpha=0.7, linestyle='--')
        
        self.crosshairs['sagittal']['h'] = self.ax_sagittal.axhline(
            y=self.cursor_z, color='lime', linewidth=1, alpha=0.7, linestyle='--')
        self.crosshairs['sagittal']['v'] = self.ax_sagittal.axvline(
            x=self.cursor_y, color='lime', linewidth=1, alpha=0.7, linestyle='--')
        
        self.crosshairs['coronal']['h'] = self.ax_coronal.axhline(
            y=self.cursor_z, color='lime', linewidth=1, alpha=0.7, linestyle='--')
        self.crosshairs['coronal']['v'] = self.ax_coronal.axvline(
            x=self.cursor_x, color='lime', linewidth=1, alpha=0.7, linestyle='--')
    
    def draw_roi(self):
        """Draw ROI rectangles"""
        for rect in self.roi_rectangles.values():
            if rect is not None:
                try:
                    rect.remove()
                except:
                    pass
        self.roi_rectangles = {}
        
        if self.roi_coords is None:
            return
        
        x_min, x_max = self.roi_coords['x']
        y_min, y_max = self.roi_coords['y']
        rect_axial = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        self.ax_axial.add_patch(rect_axial)
        self.roi_rectangles['axial'] = rect_axial
        
        y_min, y_max = self.roi_coords['y']
        z_min, z_max = self.roi_coords['z']
        rect_sagittal = Rectangle((y_min, z_min), y_max - y_min, z_max - z_min,
                                  linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        self.ax_sagittal.add_patch(rect_sagittal)
        self.roi_rectangles['sagittal'] = rect_sagittal
        
        x_min, x_max = self.roi_coords['x']
        z_min, z_max = self.roi_coords['z']
        rect_coronal = Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                                 linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        self.ax_coronal.add_patch(rect_coronal)
        self.roi_rectangles['coronal'] = rect_coronal
    
    def update_info_panel(self):
        """Update regular info panel"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        self.cursor_x = np.clip(self.cursor_x, 0, self.nx - 1)
        self.cursor_y = np.clip(self.cursor_y, 0, self.ny - 1)
        self.cursor_z = np.clip(self.cursor_z, 0, self.nz - 1)
        
        pixel_value = self.volume[self.cursor_z, self.cursor_y, self.cursor_x]
        
        roi_text = ""
        if self.roi_coords:
            roi_size = (self.roi_coords['x'][1] - self.roi_coords['x'][0] + 1,
                       self.roi_coords['y'][1] - self.roi_coords['y'][0] + 1,
                       self.roi_coords['z'][1] - self.roi_coords['z'][0] + 1)
            limit_status = "ON" if self.roi_limit_navigation else "OFF"
            roi_text = f"| ROI: {roi_size[0]}x{roi_size[1]}x{roi_size[2]} (Limit: {limit_status}) "
        
        cine_text = ""
        if self.cine_active:
            axis_names = {'x': 'Sagittal', 'y': 'Coronal', 'z': 'Axial'}
            cine_text = f"| CINE: {axis_names[self.cine_axis]} ({int(1000/self.cine_speed)} fps) "
        
        mode_text = ""
        if self.view_mode == 'organ':
            view_names = {'axial': 'Top', 'sagittal': 'Side', 'coronal': 'Front'}
            # ENHANCED: Show organ slice info
            if self.organ_slice_idx is not None:
                mode_text = f"| Organ Outlines: {view_names[self.organ_view_axis]} (Slice {self.organ_slice_idx}) "
            else:
                mode_text = f"| Organ Outlines: {view_names[self.organ_view_axis]} View "
        
        info_text = (
            f"Pos: X={self.cursor_x} Y={self.cursor_y} Z={self.cursor_z} | "
            f"Val: {pixel_value:.1f} | W/L: {self.window_width:.0f}/{self.window_center:.0f} | "
            f"Zoom: {self.zoom_level:.1f}x {roi_text}{cine_text}{mode_text}| "
            f"{self.metadata.get('PatientName', 'Unknown')} ({self.metadata.get('Modality', 'N/A')})"
        )
        
        self.ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    def export_dicom(self, event):
        """Export DICOM with FIXED orientation - correct ImageOrientationPatient values"""
        
        print("\n💾 DICOM EXPORT - Choose orientation...")
        
        popup = tk.Toplevel()
        popup.title("Export DICOM")
        popup.geometry("500x280")
        popup.configure(bg='#2b2b2b')
        popup.attributes('-topmost', True)
        
        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - 250
        y = (popup.winfo_screenheight() // 2) - 140
        popup.geometry(f'500x280+{x}+{y}')
        
        selected_orientation = {'value': None}
        
        def select_orientation(orientation):
            selected_orientation['value'] = orientation
            popup.destroy()
        
        tk.Label(popup, text="💾 Export DICOM", font=('Arial', 18, 'bold'),
                bg='#2b2b2b', fg='white').pack(pady=25)
        
        tk.Label(popup, text="Which orientation format would you like to export?",
                font=('Arial', 12), bg='#2b2b2b', fg='#cccccc').pack(pady=15)
        
        btn_frame = tk.Frame(popup, bg='#2b2b2b')
        btn_frame.pack(pady=25)
        
        tk.Button(btn_frame, text="📊 Axial", command=lambda: select_orientation('axial'),
                 font=('Arial', 13, 'bold'), bg='#4CAF50', fg='white',
                 width=11, height=2, cursor='hand2',
                 relief='raised', borderwidth=3).grid(row=0, column=0, padx=12)
        
        tk.Button(btn_frame, text="📉 Sagittal", command=lambda: select_orientation('sagittal'),
                 font=('Arial', 13, 'bold'), bg='#2196F3', fg='white',
                 width=11, height=2, cursor='hand2',
                 relief='raised', borderwidth=3).grid(row=0, column=1, padx=12)
        
        tk.Button(btn_frame, text="📈 Coronal", command=lambda: select_orientation('coronal'),
                 font=('Arial', 13, 'bold'), bg='#FF9800', fg='white',
                 width=11, height=2, cursor='hand2',
                 relief='raised', borderwidth=3).grid(row=0, column=2, padx=12)
        
        tk.Button(popup, text="✖ Cancel", command=popup.destroy,
                 font=('Arial', 11), bg='#666666', fg='white',
                 width=18, cursor='hand2').pack(pady=10)
        
        popup.wait_window()
        
        if selected_orientation['value'] is None:
            print("❌ Export cancelled by user")
            return
        
        orientation = selected_orientation['value']
        print(f"✓ Selected orientation: {orientation.upper()}")
        
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title=f"Select Export Folder for {orientation.upper()} DICOM")
        root.destroy()
        
        if not folder_path:
            print("❌ No folder selected")
            return
        
        print(f"\n💾 Exporting as {orientation.upper()}...")
        
        if self.roi_coords:
            x_range = range(self.roi_coords['x'][0], self.roi_coords['x'][1] + 1)
            y_range = range(self.roi_coords['y'][0], self.roi_coords['y'][1] + 1)
            z_range = range(self.roi_coords['z'][0], self.roi_coords['z'][1] + 1)
        else:
            x_range = range(self.nx)
            y_range = range(self.ny)
            z_range = range(self.nz)
        
        roi_volume = self.volume[
            min(z_range):max(z_range)+1,
            min(y_range):max(y_range)+1,
            min(x_range):max(x_range)+1
        ]
        
        print(f"ROI Volume shape: {roi_volume.shape} (Z, Y, X)")
        
        # FIXED: Correct volume orientation and axis mapping
        if orientation == 'axial':
            # Axial: Keep original (Z, Y, X)
            export_volume = roi_volume
            num_slices = roi_volume.shape[0]
            slice_axis = 'Z'
            print(f"Axial export: {num_slices} slices along Z-axis")
            
        elif orientation == 'sagittal':
            # Sagittal: Transpose to (X, Z, Y) - slicing along X
            export_volume = np.transpose(roi_volume, (2, 0, 1))
            num_slices = export_volume.shape[0]
            slice_axis = 'X'
            print(f"Sagittal export: {num_slices} slices along X-axis")
            
        elif orientation == 'coronal':
            # Coronal: Transpose to (Y, Z, X) - slicing along Y
            export_volume = np.transpose(roi_volume, (1, 0, 2))
            num_slices = export_volume.shape[0]
            slice_axis = 'Y'
            print(f"Coronal export: {num_slices} slices along Y-axis")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i in range(num_slices):
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            file_meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.3680043.8.498.{timestamp}.{i}"
            file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
            file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.1'
            
            ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
            
            if self.dicom_files and len(self.dicom_files) > 0:
                original_ds = self.dicom_files[0]
                for attr in ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
                            'StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription',
                            'Modality', 'Manufacturer', 'InstitutionName']:
                    if hasattr(original_ds, attr):
                        setattr(ds, attr, getattr(original_ds, attr))
            else:
                ds.PatientName = self.metadata.get('PatientName', 'Unknown')
                ds.PatientID = "EXPORT"
                ds.Modality = self.metadata.get('Modality', 'CT')
            
            ds.SeriesInstanceUID = f"1.2.826.0.1.3680043.8.498.{timestamp}.{orientation}"
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.InstanceNumber = i + 1
            ds.SeriesNumber = 9000 + {'axial': 1, 'sagittal': 2, 'coronal': 3}[orientation]
            ds.SeriesDescription = f"Exported {orientation.capitalize()} View"
            
            slice_data = export_volume[i, :, :]
            slice_data_int = slice_data.astype(np.int16)
            
            ds.Rows = slice_data_int.shape[0]
            ds.Columns = slice_data_int.shape[1]
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            
            # FIXED: Correct ImageOrientationPatient values
            # Format: [row_x, row_y, row_z, col_x, col_y, col_z]
            # Defines direction cosines of first row and first column
            
            if orientation == 'axial':
                # Axial: X-axis is row direction, Y-axis is column direction
                # Looking from feet to head (standard axial view)
                ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                # ImagePositionPatient: X, Y, Z position of upper-left pixel
                ds.ImagePositionPatient = [0.0, 0.0, float(i)]
                ds.SliceLocation = float(i)
                
            elif orientation == 'sagittal':
                # Sagittal: Y-axis is row direction, Z-axis is column direction (flipped)
                # Looking from right to left side of patient
                ds.ImageOrientationPatient = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                # ImagePositionPatient: X position changes with slice
                ds.ImagePositionPatient = [float(i), 0.0, float(ds.Rows - 1)]
                ds.SliceLocation = float(i)
                
            elif orientation == 'coronal':
                # Coronal: X-axis is row direction, Z-axis is column direction (flipped)
                # Looking from front to back of patient
                ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                # ImagePositionPatient: Y position changes with slice
                ds.ImagePositionPatient = [0.0, float(i), float(ds.Rows - 1)]
                ds.SliceLocation = float(i)
            
            ds.PixelData = slice_data_int.tobytes()
            
            filename = os.path.join(folder_path, f"{orientation}_slice_{i:04d}.dcm")
            ds.save_as(filename, write_like_original=False)
            
            if i == 0:
                print(f"\n✓ First file metadata:")
                print(f"  ImageOrientationPatient: {ds.ImageOrientationPatient}")
                print(f"  ImagePositionPatient: {ds.ImagePositionPatient}")
                print(f"  Slice Location: {ds.SliceLocation}")
                print(f"  Rows: {ds.Rows}, Columns: {ds.Columns}")
        
        success_popup = tk.Tk()
        success_popup.withdraw()
        success_popup.attributes('-topmost', True)
        messagebox.showinfo("Export Complete", 
                           f"✓ Successfully exported {num_slices} DICOM files\n\n"
                           f"Orientation: {orientation.upper()}\n"
                           f"Slice Axis: {slice_axis}\n\n"
                           f"ImageOrientationPatient:\n{ds.ImageOrientationPatient}\n\n"
                           f"Location:\n{folder_path}")
        success_popup.destroy()
        
        print(f"\n✅ EXPORT COMPLETE!")
        print(f"✓ Exported {num_slices} slices as {orientation.upper()}")
        print(f"✓ Slice axis: {slice_axis}")
        print(f"✓ ImageOrientationPatient: {ds.ImageOrientationPatient}")
        print(f"✓ Location: {folder_path}\n")
        print(f"🔄 Re-upload this folder to verify correct orientation!\n")
         
    def on_click(self, event):
        """Handle clicks"""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        
        roi_selector_active = False
        if hasattr(self, 'roi_selector_axial'):
            if (self.roi_selector_axial.active or 
                self.roi_selector_sagittal.active or 
                self.roi_selector_coronal.active):
                roi_selector_active = True
        
        if roi_selector_active or self.roi_drawing_mode:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        limits = self.get_roi_limits()
        
        if event.button == 2:
            if event.inaxes == self.ax_axial:
                self.pan_active = 'axial'
            elif event.inaxes == self.ax_sagittal:
                self.pan_active = 'sagittal'
            elif event.inaxes == self.ax_coronal:
                self.pan_active = 'coronal'
            elif event.inaxes == self.ax_4th:
                self.pan_active = '4th'
            
            if self.pan_active:
                self.pan_start = (event.xdata, event.ydata)
            return
        
        if event.inaxes == self.ax_axial:
            if event.button == 1:
                self.cursor_x = np.clip(x, limits['x'][0], limits['x'][1])
                self.cursor_y = np.clip(y, limits['y'][0], limits['y'][1])
                self.update_in_progress = True
                self.slice_x_slider.set_val(self.cursor_x)
                self.slice_y_slider.set_val(self.cursor_y)
                self.update_in_progress = False
            elif event.button == 3:
                self.oblique_center_x = np.clip(x, 0, self.nx - 1)
                self.oblique_center_y = np.clip(y, 0, self.ny - 1)
                self.oblique_cache = None
        
        elif event.inaxes == self.ax_sagittal:
            if event.button == 1:
                self.cursor_y = np.clip(x, limits['y'][0], limits['y'][1])
                self.cursor_z = np.clip(y, limits['z'][0], limits['z'][1])
                self.update_in_progress = True
                self.slice_y_slider.set_val(self.cursor_y)
                self.slice_z_slider.set_val(self.cursor_z)
                self.update_in_progress = False
            elif event.button == 3:
                self.oblique_center_y = np.clip(x, 0, self.ny - 1)
                self.oblique_center_z = np.clip(y, 0, self.nz - 1)
                self.oblique_cache = None
        
        elif event.inaxes == self.ax_coronal:
            if event.button == 1:
                self.cursor_x = np.clip(x, limits['x'][0], limits['x'][1])
                self.cursor_z = np.clip(y, limits['z'][0], limits['z'][1])
                self.update_in_progress = True
                self.slice_x_slider.set_val(self.cursor_x)
                self.slice_z_slider.set_val(self.cursor_z)
                self.update_in_progress = False
            elif event.button == 3:
                self.oblique_center_x = np.clip(x, 0, self.nx - 1)
                self.oblique_center_z = np.clip(y, 0, self.nz - 1)
                self.oblique_cache = None
        
        self.update_all_views()
        self.update_info_panel()
    
    def on_release(self, event):
        """Handle mouse release"""
        if event.button == 2:
            self.pan_active = None
            self.pan_start = None
    
    def on_motion(self, event):
        """Handle mouse motion for panning"""
        if self.pan_active and self.pan_start and event.xdata and event.ydata:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            self.pan_offsets[self.pan_active][0] += dx
            self.pan_offsets[self.pan_active][1] += dy
            
            self.pan_start = (event.xdata, event.ydata)
            self.update_all_views()
    
    def on_scroll(self, event):
        """Handle scroll - ENHANCED with organ outline slice scrolling"""
        limits = self.get_roi_limits()
        
        if event.inaxes == self.ax_axial:
            if event.button == 'up':
                self.cursor_z = min(self.cursor_z + 1, limits['z'][1])
            else:
                self.cursor_z = max(self.cursor_z - 1, limits['z'][0])
            self.update_in_progress = True
            self.slice_z_slider.set_val(self.cursor_z)
            self.update_in_progress = False
            self.update_all_views()
            self.update_info_panel()
        
        elif event.inaxes == self.ax_sagittal:
            if event.button == 'up':
                self.cursor_x = min(self.cursor_x + 1, limits['x'][1])
            else:
                self.cursor_x = max(self.cursor_x - 1, limits['x'][0])
            self.update_in_progress = True
            self.slice_x_slider.set_val(self.cursor_x)
            self.update_in_progress = False
            self.update_all_views()
            self.update_info_panel()
        
        elif event.inaxes == self.ax_coronal:
            if event.button == 'up':
                self.cursor_y = min(self.cursor_y + 1, limits['y'][1])
            else:
                self.cursor_y = max(self.cursor_y - 1, limits['y'][0])
            self.update_in_progress = True
            self.slice_y_slider.set_val(self.cursor_y)
            self.update_in_progress = False
            self.update_all_views()
            self.update_info_panel()
        
        # ENHANCED: Scroll through organ outline slices in 4th view
        elif event.inaxes == self.ax_4th and self.view_mode == 'organ':
            if self.organ_volume is not None and self.organ_slice_idx is not None:
                organ_shape = self.organ_volume.shape
                
                # Get max slice based on current axis
                if self.organ_view_axis == 'axial':
                    max_slice = organ_shape[0] - 1
                elif self.organ_view_axis == 'sagittal':
                    max_slice = organ_shape[2] - 1 if len(organ_shape) > 2 else organ_shape[0] - 1
                elif self.organ_view_axis == 'coronal':
                    max_slice = organ_shape[1] - 1 if len(organ_shape) > 1 else organ_shape[0] - 1
                
                # Update slice index
                if event.button == 'up':
                    self.organ_slice_idx = min(self.organ_slice_idx + 1, max_slice)
                else:
                    self.organ_slice_idx = max(self.organ_slice_idx - 1, 0)
                
                print(f"🎯 Organ Outlines: Slice {self.organ_slice_idx}/{max_slice} ({self.organ_view_axis})")
                self.update_all_views()
                self.update_info_panel()
        
        else:
            if event.button == 'up':
                self.adjust_zoom(1.1)
            else:
                self.adjust_zoom(0.9)
    
    def on_key(self, event):
        """Handle keyboard shortcuts - ENHANCED with organ outline slice navigation"""
        changed = False
        limits = self.get_roi_limits()
        
        if event.key == 'up':
            self.cursor_z = min(self.cursor_z + 1, limits['z'][1])
            self.update_in_progress = True
            self.slice_z_slider.set_val(self.cursor_z)
            self.update_in_progress = False
            changed = True
        elif event.key == 'down':
            self.cursor_z = max(self.cursor_z - 1, limits['z'][0])
            self.update_in_progress = True
            self.slice_z_slider.set_val(self.cursor_z)
            self.update_in_progress = False
            changed = True
        elif event.key == 'w':
            self.cursor_y = min(self.cursor_y + 1, limits['y'][1])
            self.update_in_progress = True
            self.slice_y_slider.set_val(self.cursor_y)
            self.update_in_progress = False
            changed = True
        elif event.key == 's':
            self.cursor_y = max(self.cursor_y - 1, limits['y'][0])
            self.update_in_progress = True
            self.slice_y_slider.set_val(self.cursor_y)
            self.update_in_progress = False
            changed = True
        elif event.key == 'a':
            self.cursor_x = max(self.cursor_x - 1, limits['x'][0])
            self.update_in_progress = True
            self.slice_x_slider.set_val(self.cursor_x)
            self.update_in_progress = False
            changed = True
        elif event.key == 'd':
            self.cursor_x = min(self.cursor_x + 1, limits['x'][1])
            self.update_in_progress = True
            self.slice_x_slider.set_val(self.cursor_x)
            self.update_in_progress = False
            changed = True
        elif event.key == 'left':
            self.oblique_rotation_xy -= 5
            self.oblique_xy_slider.set_val(self.oblique_rotation_xy)
            changed = True
        elif event.key == 'right':
            self.oblique_rotation_xy += 5
            self.oblique_xy_slider.set_val(self.oblique_rotation_xy)
            changed = True
        elif event.key == 'pageup':
            self.oblique_rotation_z = min(self.oblique_rotation_z + 5, 90)
            self.oblique_z_slider.set_val(self.oblique_rotation_z)
            changed = True
        elif event.key == 'pagedown':
            self.oblique_rotation_z = max(self.oblique_rotation_z - 5, -90)
            self.oblique_z_slider.set_val(self.oblique_rotation_z)
            changed = True
        
        # ENHANCED: Keyboard shortcuts for organ outline slice navigation
        elif event.key == '[' and self.view_mode == 'organ':
            # Previous organ outline slice
            if self.organ_volume is not None and self.organ_slice_idx is not None:
                self.organ_slice_idx = max(self.organ_slice_idx - 1, 0)
                organ_shape = self.organ_volume.shape
                if self.organ_view_axis == 'axial':
                    max_slice = organ_shape[0] - 1
                elif self.organ_view_axis == 'sagittal':
                    max_slice = organ_shape[2] - 1 if len(organ_shape) > 2 else organ_shape[0] - 1
                elif self.organ_view_axis == 'coronal':
                    max_slice = organ_shape[1] - 1 if len(organ_shape) > 1 else organ_shape[0] - 1
                print(f"🎯 Organ Outlines: Slice {self.organ_slice_idx}/{max_slice} ({self.organ_view_axis})")
                changed = True
        
        elif event.key == ']' and self.view_mode == 'organ':
            # Next organ outline slice
            if self.organ_volume is not None and self.organ_slice_idx is not None:
                organ_shape = self.organ_volume.shape
                if self.organ_view_axis == 'axial':
                    max_slice = organ_shape[0] - 1
                elif self.organ_view_axis == 'sagittal':
                    max_slice = organ_shape[2] - 1 if len(organ_shape) > 2 else organ_shape[0] - 1
                elif self.organ_view_axis == 'coronal':
                    max_slice = organ_shape[1] - 1 if len(organ_shape) > 1 else organ_shape[0] - 1
                
                self.organ_slice_idx = min(self.organ_slice_idx + 1, max_slice)
                print(f"🎯 Organ Outlines: Slice {self.organ_slice_idx}/{max_slice} ({self.organ_view_axis})")
                changed = True
        
        elif event.key == 'r':
            self.reset_window_level(None)
            changed = True
        elif event.key == 'm' and self.organ_volume is not None:
            self.toggle_view_mode(None)
            changed = True
        elif event.key == '+' or event.key == '=':
            self.adjust_zoom(1.2)
            changed = True
        elif event.key == '-':
            self.adjust_zoom(0.8)
            changed = True
        elif event.key == 'c':
            self.toggle_cine(None)
        elif event.key == 'x':
            self.cine_axis = 'x'
            self.cine_axis_btn.label.set_text('Cine:\nSagittal')
        elif event.key == 'y':
            self.cine_axis = 'y'
            self.cine_axis_btn.label.set_text('Cine:\nCoronal')
        elif event.key == 'z':
            self.cine_axis = 'z'
            self.cine_axis_btn.label.set_text('Cine:\nAxial')
        elif event.key == 'p':
            self.reset_pan(None)
        elif event.key == 'e':
            if hasattr(self, 'roi_selector_axial') and self.roi_selector_axial.active:
                self.activate_roi_selection(None)
        elif event.key == 'q':
            if self.cine_active:
                self.stop_cine()
            if self.ai_organ_detector:
                self.ai_organ_detector.cleanup()
            plt.close()
            return
        
        if changed:
            self.update_all_views()
            self.update_info_panel()
        
    def load_dicom_dialog(self, event):
        """Load DICOM with DUAL AI detection"""
        root = Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select DICOM Folder")
        root.destroy()
        
        if folder_path:
            print("Loading DICOM...")
            volume, metadata, dicom_files = load_dicom_series(folder_path)
            if volume is not None:
                self.volume = volume
                self.metadata = metadata
                self.dicom_files = dicom_files
                self.dicom_path = folder_path
                self.nz, self.ny, self.nx = volume.shape
                
                self.cursor_x = self.nx // 2
                self.cursor_y = self.ny // 2
                self.cursor_z = self.nz // 2
                
                self.roi_coords = None
                self.roi_limit_navigation = False
                self.roi_drawing_mode = False
                self.roi_limit_btn.label.set_text('Limit:\nOFF')
                self.roi_limit_btn.color = 'orange'
                
                self.pan_offsets = {'axial': [0, 0], 'sagittal': [0, 0], 'coronal': [0, 0], '4th': [0, 0]}
                
                self.update_in_progress = True
                self.slice_x_slider.valmin = 0
                self.slice_x_slider.valmax = max(0, self.nx - 1)
                self.slice_y_slider.valmin = 0
                self.slice_y_slider.valmax = max(0, self.ny - 1)
                self.slice_z_slider.valmin = 0
                self.slice_z_slider.valmax = max(0, self.nz - 1)
                self.slice_x_slider.set_val(self.cursor_x)
                self.slice_y_slider.set_val(self.cursor_y)
                self.slice_z_slider.set_val(self.cursor_z)
                self.update_in_progress = False
                
                self.window_center = np.median(volume)
                self.window_width = np.percentile(volume, 95) - np.percentile(volume, 5)
                self.default_center = self.window_center
                self.default_width = self.window_width
                
                self.window_slider.valmin = self.window_width * 0.1
                self.window_slider.valmax = self.window_width * 3
                self.level_slider.valmin = np.min(volume)
                self.level_slider.valmax = np.max(volume)
                self.window_slider.set_val(self.window_width)
                self.level_slider.set_val(self.window_center)
                
                self.oblique_cache = None
                
                print(f"✓ Loaded: {self.nx} x {self.ny} x {self.nz}")
                self.update_all_views()
                self.update_info_panel()
                
                if HAS_ORGAN_DETECTOR or HAS_ORIENTATION_DETECTOR:
                    self.start_ai_detection()
                else:
                    self.update_ai_panel("⚠ AI Detection not available")
    
    def load_nifti_dialog(self, event):
        """Load NIfTI - ENHANCED with organ slice initialization"""
        if not HAS_NIBABEL:
            print("✗ ERROR: nibabel required")
            return
        
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select NIfTI File",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        )
        root.destroy()
        
        if file_path:
            print("Loading NIfTI...")
            organ_volume = load_nii_file(file_path)
            if organ_volume is not None:
                self.organ_volume = organ_volume
                self.detect_organ_orientation()
                
                # ENHANCED: Initialize organ slice to middle
                organ_shape = organ_volume.shape
                self.organ_slice_idx = organ_shape[0] // 2  # Default to axial middle
                
                print(f"✓ Loaded organ: {organ_volume.shape}")
                print(f"✓ Organ slice initialized to: {self.organ_slice_idx}")
                
                if not hasattr(self, 'mode_btn'):
                    self.ax_mode_btn = self.fig.add_axes([0.90, 0.005, 0.09, 0.028])
                    self.mode_btn = Button(self.ax_mode_btn, 'Mode:\nOblique', color='lightgray', hovercolor='silver')
                    self.mode_btn.on_clicked(self.toggle_view_mode)
                    
                    self.ax_organ_axis = self.fig.add_axes([0.90, 0.04, 0.09, 0.08])
                    self.organ_axis_radio = RadioButtons(self.ax_organ_axis, ('Axial', 'Sagittal', 'Coronal'))
                    self.organ_axis_radio.on_clicked(self.set_organ_axis)
                    self.ax_organ_axis.set_visible(False)
                
                self.update_all_views()
    
    def activate_roi_selection(self, event):
        """Activate ROI selection - Toggle mode"""
        if hasattr(self, 'roi_selector_axial') and self.roi_selector_axial.active:
            self.roi_selector_axial.set_active(False)
            self.roi_selector_sagittal.set_active(False)
            self.roi_selector_coronal.set_active(False)
            self.roi_active = False
            self.roi_drawing_mode = False
            self.roi_btn.label.set_text('Define\nROI')
            self.roi_btn.color = 'yellow'
            print("✓ ROI selection mode EXITED")
            return
        
        print("\n🔲 ROI SELECTION MODE - Draw rectangle, then click button again to exit")
        
        self.roi_active = True
        self.roi_drawing_mode = False
        self.roi_btn.label.set_text('Exit ROI\nMode')
        self.roi_btn.color = 'orange'
        
        def on_select_start():
            self.roi_drawing_mode = True
        
        def on_select_axial(eclick, erelease):
            self.roi_drawing_mode = False
            x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
            y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])
            
            if self.roi_coords and 'z' in self.roi_coords:
                z_bounds = self.roi_coords['z']
            else:
                z_bounds = (0, self.nz - 1)
            
            self.roi_coords = {
                'x': (max(0, x1), min(self.nx - 1, x2)),
                'y': (max(0, y1), min(self.ny - 1, y2)),
                'z': z_bounds
            }
            print(f"✓ ROI defined from AXIAL")
            self.update_all_views()
        
        def on_select_sagittal(eclick, erelease):
            self.roi_drawing_mode = False
            y1, y2 = sorted([int(eclick.xdata), int(erelease.xdata)])
            z1, z2 = sorted([int(eclick.ydata), int(erelease.ydata)])
            
            if self.roi_coords and 'x' in self.roi_coords:
                x_bounds = self.roi_coords['x']
            else:
                x_bounds = (0, self.nx - 1)
            
            self.roi_coords = {
                'x': x_bounds,
                'y': (max(0, y1), min(self.ny - 1, y2)),
                'z': (max(0, z1), min(self.nz - 1, z2))
            }
            print(f"✓ ROI defined from SAGITTAL")
            self.update_all_views()
        
        def on_select_coronal(eclick, erelease):
            self.roi_drawing_mode = False
            x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
            z1, z2 = sorted([int(eclick.ydata), int(erelease.ydata)])
            
            if self.roi_coords and 'y' in self.roi_coords:
                y_bounds = self.roi_coords['y']
            else:
                y_bounds = (0, self.ny - 1)
            
            self.roi_coords = {
                'x': (max(0, x1), min(self.nx - 1, x2)),
                'y': y_bounds,
                'z': (max(0, z1), min(self.nz - 1, z2))
            }
            print(f"✓ ROI defined from CORONAL")
            self.update_all_views()
        
        self.roi_selector_axial = RectangleSelector(
            self.ax_axial, on_select_axial, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        self.roi_selector_sagittal = RectangleSelector(
            self.ax_sagittal, on_select_sagittal, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        self.roi_selector_coronal = RectangleSelector(
            self.ax_coronal, on_select_coronal, useblit=True, button=[1],
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
    
    def clear_roi(self, event):
        """Clear ROI"""
        self.roi_active = False
        self.roi_drawing_mode = False
        self.roi_coords = None
        self.roi_limit_navigation = False
        self.roi_limit_btn.label.set_text('Limit:\nOFF')
        self.roi_limit_btn.color = 'orange'
        
        for rect in self.roi_rectangles.values():
            if rect is not None:
                try:
                    rect.remove()
                except:
                    pass
        self.roi_rectangles = {}
        
        if hasattr(self, 'roi_selector_axial'):
            self.roi_selector_axial.set_active(False)
            self.roi_selector_sagittal.set_active(False)
            self.roi_selector_coronal.set_active(False)
        
        self.roi_btn.label.set_text('Define\nROI')
        self.roi_btn.color = 'yellow'
        
        print("✓ ROI cleared")
        self.update_all_views()
        self.update_info_panel()
    
    def toggle_cine(self, event):
        if self.cine_active:
            self.stop_cine()
        else:
            self.start_cine()
    
    def start_cine(self):
        self.cine_active = True
        self.cine_btn.label.set_text('Stop\nCine')
        self.cine_step()
    
    def stop_cine(self):
        self.cine_active = False
        self.cine_btn.label.set_text('Start\nCine')
        if self.cine_timer is not None:
            self.cine_timer.stop()
            self.cine_timer = None
        self.update_info_panel()
    
    def cine_step(self):
        if not self.cine_active:
            return
        
        limits = self.get_roi_limits()
        
        if self.cine_axis == 'z':
            self.cursor_z += self.cine_direction
            if self.cursor_z >= limits['z'][1]:
                self.cursor_z = limits['z'][1]
                self.cine_direction = -1
            elif self.cursor_z <= limits['z'][0]:
                self.cursor_z = limits['z'][0]
                self.cine_direction = 1
            self.update_in_progress = True
            self.slice_z_slider.set_val(self.cursor_z)
            self.update_in_progress = False
        elif self.cine_axis == 'y':
            self.cursor_y += self.cine_direction
            if self.cursor_y >= limits['y'][1]:
                self.cursor_y = limits['y'][1]
                self.cine_direction = -1
            elif self.cursor_y <= limits['y'][0]:
                self.cursor_y = limits['y'][0]
                self.cine_direction = 1
            self.update_in_progress = True
            self.slice_y_slider.set_val(self.cursor_y)
            self.update_in_progress = False
        elif self.cine_axis == 'x':
            self.cursor_x += self.cine_direction
            if self.cursor_x >= limits['x'][1]:
                self.cursor_x = limits['x'][1]
                self.cine_direction = -1
            elif self.cursor_x <= limits['x'][0]:
                self.cursor_x = limits['x'][0]
                self.cine_direction = 1
            self.update_in_progress = True
            self.slice_x_slider.set_val(self.cursor_x)
            self.update_in_progress = False
        
        self.update_all_views()
        self.update_info_panel()
        
        self.cine_timer = self.fig.canvas.new_timer(interval=self.cine_speed)
        self.cine_timer.single_shot = True
        self.cine_timer.add_callback(self.cine_step)
        self.cine_timer.start()
    
    def show(self):
        """Show viewer"""
        print("\n" + "="*70)
        print("🏥 MPV MEDICAL VIEWER - DUAL AI ENHANCED")
        print("="*70)
        print("\n🤖 DUAL AI Features:")
        print("  ✓ Organ Detection (TotalSegmentator)")
        print("  ✓ Orientation Detection (3D CNN ResNet18 - 21 conv layers)")
        print("  ✓ Shows Top 3 organs + Orientation in info panel")
        print("  ✓ GPU-accelerated")
        print("\n📍 Enhanced Features:")
        print("  ✓ Navigation, ROI, Cine, Export with orientation choice")
        print("  ✓ Oblique plane visualization with yellow lines")
        print("  ✓ FIXED: Correct ImageOrientationPatient for exports")
        print("  ✓ ENHANCED NIfTI organ OUTLINE viewer with:")
        print("    • Multi-colored contours for different organs")
        print("    • Sobel edge detection for clean outlines")
        print("    • Independent slice scrolling (mouse wheel on 4th view)")
        print("    • Keyboard shortcuts: [ ] keys to navigate slices")
        print("    • Slice counter displayed in title")
        print("    • 'hot' colormap for vibrant organ boundaries")
        print("\n🔧 EXPORT FIX:")
        print("  ✓ Axial: [1,0,0, 0,1,0] - X-row, Y-col")
        print("  ✓ Sagittal: [0,1,0, 0,0,-1] - Y-row, Z-col (flipped)")
        print("  ✓ Coronal: [1,0,0, 0,0,-1] - X-row, Z-col (flipped)")
        print("="*70 + "\n")
        plt.show()


# Helper functions outside the class
def load_dicom_series(folder_path):
    """Load DICOM series"""
    folder = Path(folder_path)
    if not folder.is_dir():
        return None, None, None
    
    dicom_files = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in ['.dcm', '']:
            try:
                ds = pydicom.dcmread(str(file), force=True)
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append(ds)
            except:
                pass
    
    if not dicom_files:
        return None, None, None
    
    try:
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        try:
            dicom_files.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass
    
    slice_arrays = []
    for ds in dicom_files:
        pixel_array = ds.pixel_array.astype(float)
        if hasattr(ds, 'RescaleSlope'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        slice_arrays.append(pixel_array)
    
    volume = np.stack(slice_arrays, axis=0)
    metadata = {
        'PatientName': str(dicom_files[0].PatientName) if hasattr(dicom_files[0], 'PatientName') else 'Unknown',
        'Modality': str(dicom_files[0].Modality) if hasattr(dicom_files[0], 'Modality') else 'N/A',
    }
    
    return volume, metadata, dicom_files


def load_nii_file(nii_path):
    """Load NIfTI file"""
    if not HAS_NIBABEL:
        return None
    img = nib.load(nii_path)
    data = img.get_fdata()
    return data


def main():
    print("\n" + "="*70)
    print("🏥 MPV MEDICAL VIEWER - DUAL AI ENHANCED (FIXED EXPORT)")
    print(f"👤 User: mazenElzeiny")
    print(f"📅 Date: 2025-10-21 21:24:13 UTC")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\nUsage: python mpv_viewer.py <dicom_folder> [nii_file]")
        print("Or use GUI buttons\n")
        volume = np.random.rand(100, 256, 256) * 1000
        metadata = {'PatientName': 'Demo', 'Modality': 'CT'}
        organ_volume = None
        dicom_files = None
        dicom_path = None
    else:
        dicom_path = sys.argv[1]
        volume, metadata, dicom_files = load_dicom_series(dicom_path)
        if volume is None:
            sys.exit(1)
        organ_volume = None
        if len(sys.argv) > 2:
            organ_volume = load_nii_file(sys.argv[2])
    
    print(f"\n📋 Shape: {volume.shape}")
    for k, v in metadata.items():
        print(f"   {k}: {v}")
    
    viewer = MPVViewer(volume, metadata, organ_volume, dicom_files, dicom_path)
    viewer.show()


if __name__ == "__main__":
    main()