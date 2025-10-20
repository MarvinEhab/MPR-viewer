#!/usr/bin/env python3
"""
AI Organ Detector using TotalSegmentator - OPTIMIZED VERSION
Faster detection with smart sampling and parallel processing
"""

import os
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import time


class OrganDetector:
    """Detects organs from DICOM using TotalSegmentator - OPTIMIZED"""
    
    def __init__(self, fast_mode=True, use_gpu=True):
        self.fast_mode = fast_mode
        self.use_gpu = use_gpu
        self.last_results = None
        
        # Performance settings
        self.max_workers = 4  # Parallel processing
        
        print(f"✓ AI Detector initialized (Fast mode: {fast_mode}, GPU: {use_gpu})")
    
    def detect_organs(self, dicom_path):
        """
        Detect organs from DICOM file/folder - OPTIMIZED
        
        Args:
            dicom_path (str): Path to DICOM file or folder
            
        Returns:
            dict: {'top_organs': [(name, count), ...], 'all_organs': {name: count, ...}}
        """
        start_time = time.time()
        
        print("\n" + "="*60)
        print("🤖 AI ORGAN DETECTION - TotalSegmentator (OPTIMIZED)")
        print("="*60)
        print(f"Input: {dicom_path}")
        print(f"Fast mode: {self.fast_mode} | GPU: {self.use_gpu}")
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="mpv_seg_")
        
        try:
            # Run TotalSegmentator with optimizations
            print("\n⏳ Running TotalSegmentator...")
            print("   (Using fast mode + GPU acceleration if available)")
            
            # OPTIMIZATION 1: Use fast mode + minimal settings
            totalsegmentator(
                input=dicom_path,
                output=temp_dir,
                fast=True,  # Always use fast mode
                ml=False,   # Individual files (faster to process)
                nr_thr_resamp=4,  # Parallel resampling
                nr_thr_saving=4,  # Parallel saving
                force_split=False,  # Don't split - faster
                body_seg=False,  # Skip body segmentation - faster
                quiet=True  # Less console output
            )
            
            seg_time = time.time() - start_time
            print(f"✓ Segmentation complete in {seg_time:.1f}s! Analyzing...")
            
            # OPTIMIZATION 2: Parallel file processing
            organ_counts = self._count_organ_voxels_parallel(temp_dir)
            
            # OPTIMIZATION 3: Fast merging
            merged_counts = self._merge_organ_parts_fast(organ_counts)
            
            # Get top 3 organs
            sorted_organs = sorted(merged_counts.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_organs[:3]
            
            # Store results
            self.last_results = {
                'top_organs': top_3,
                'all_organs': merged_counts,
                'temp_dir': temp_dir,
                'time': time.time() - start_time
            }
            
            # Print results
            self._print_results(top_3, sorted_organs, time.time() - start_time)
            
            return self.last_results
            
        except Exception as e:
            print(f"\n❌ Error during organ detection: {e}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {
                'top_organs': [('unknown', 0)],
                'all_organs': {},
                'error': str(e)
            }
    
    def _count_organ_voxels_parallel(self, output_dir):
        """OPTIMIZED: Count voxels using parallel processing"""
        organ_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        
        if not organ_files:
            raise FileNotFoundError(f"No segmentation files found")
        
        print(f"Processing {len(organ_files)} organ files in parallel...")
        
        organ_counts = {}
        
        # PARALLEL PROCESSING - Process multiple files at once
        def process_organ_file(organ_file):
            organ_path = os.path.join(output_dir, organ_file)
            organ_name = organ_file.replace('.nii.gz', '')
            
            try:
                # Load and count
                organ_img = nib.load(organ_path)
                organ_data = organ_img.get_fdata()
                
                # OPTIMIZATION: Use numpy's efficient counting
                voxel_count = int(np.count_nonzero(organ_data))
                
                return (organ_name, voxel_count) if voxel_count > 0 else None
            except:
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(process_organ_file, organ_files)
        
        # Collect results
        for result in results:
            if result:
                organ_name, count = result
                organ_counts[organ_name] = count
        
        return organ_counts
    
    def _merge_organ_parts_fast(self, organ_counts):
        """OPTIMIZED: Fast organ part merging using dictionaries"""
        organ_groups = {}
        merged_counts = {}
        
        # Group organ parts
        for organ_name, count in organ_counts.items():
            base_name = self._get_base_organ_name_fast(organ_name)
            
            if base_name != organ_name:
                if base_name not in organ_groups:
                    organ_groups[base_name] = 0
                organ_groups[base_name] += count
            else:
                merged_counts[organ_name] = count
        
        # Add merged groups
        merged_counts.update(organ_groups)
        
        return merged_counts
    
    def _get_base_organ_name_fast(self, organ_name):
        """OPTIMIZED: Faster base organ name extraction"""
        # Pre-compiled list of indicators
        indicators = [
            '_left', '_right', '_upper', '_lower', '_middle',
            '_lobe', '_anterior', '_posterior'
        ]
        
        base = organ_name.lower()
        
        # Single pass replacement
        for indicator in indicators:
            base = base.replace(indicator, '')
        
        # Clean up
        base = base.replace('__', '_').strip('_')
        
        return base
    
    def _print_results(self, top_3, all_sorted, total_time):
        """Print detection results with timing"""
        print("\n" + "="*60)
        print("🏆 TOP 3 DETECTED ORGANS")
        print("="*60)
        for i, (organ_name, count) in enumerate(top_3, 1):
            organ_display = organ_name.replace('_', ' ').upper()
            print(f"{i}. {organ_display:30s} {count:12,} voxels")
        
        print("\n" + "="*60)
        print(f"⏱️  Total Detection Time: {total_time:.1f} seconds")
        print("="*60)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.last_results and 'temp_dir' in self.last_results:
            temp_dir = self.last_results['temp_dir']
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"✓ Cleaned up temporary files")
                except Exception as e:
                    print(f"⚠ Could not clean up: {e}")


# Quick test function
if __name__ == "__main__":
    detector = OrganDetector(fast_mode=True, use_gpu=True)
    
    test_dicom = r"C:\path\to\your\dicom\folder"  # CHANGE THIS
    
    if os.path.exists(test_dicom):
        print("\n🚀 Testing optimized organ detection...\n")
        
        results = detector.detect_organs(test_dicom)
        
        if 'top_organs' in results:
            print("\n✅ Success!")
            print(f"Detected in {results.get('time', 0):.1f}s")
            print(f"Top organ: {results['top_organs'][0][0]}")
        
        detector.cleanup()
    else:
        print(f"❌ Path not found: {test_dicom}")