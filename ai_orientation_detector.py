#!/usr/bin/env python3
"""
AI Orientation Detector - EXACT MATCH for trained model
Detects DICOM orientation: Axial, Frontal (Coronal), Sagittal
Matches training preprocessing exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydicom
from pathlib import Path
import os
from PIL import Image


class BasicBlock(nn.Module):
    """ResNet BasicBlock with optional downsample"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet18 for 2D images"""
    
    def __init__(self, block, layers, num_classes=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial conv layer - MODIFIED FOR 1 CHANNEL INPUT
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes=3):
    """ResNet18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


class DicomOrientationClassifier(nn.Module):
    """Wrapper matching training architecture"""
    def __init__(self, num_classes=3):
        super(DicomOrientationClassifier, self).__init__()
        self.backbone = resnet18(num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class OrientationDetector:
    """Detects orientation from DICOM - EXACT MATCH for training"""
    
    def __init__(self, model_path='best_orientation_classifier.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # IMPORTANT: Class order MUST match training
        # Training used: ['axial', 'frontal', 'sagittal'] alphabetically
        self.orientations = {
            0: 'axial',
            1: 'coronal',  # 'frontal' in training = coronal view
            2: 'sagittal'
        }
        
        # Training parameters - MUST MATCH dataset.py
        self.window_center = 40
        self.window_width = 400
        self.img_size = 224
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        
        self.last_result = None
        
        # Check if model file exists
        if os.path.exists(model_path):
            self._load_model()
            print(f"✓ Orientation Detector (ResNet18) loaded from {model_path}")
        else:
            print(f"⚠ Model file not found: {model_path}")
    
    def _load_model(self):
        """Load trained ResNet18 model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model = DicomOrientationClassifier(num_classes=3)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            val_acc = checkpoint.get('val_acc', 0)
            print(f"  Model accuracy: {val_acc:.2f}%")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def apply_windowing(self, image, center, width):
        """Apply windowing (level and width) to DICOM image - MATCHES training"""
        img_min = center - width / 2
        img_max = center + width / 2
        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min)
        return image
    
    def preprocess_dicom_slice(self, pixel_array, rescale_slope=1.0, rescale_intercept=0.0):
        """
        Preprocess single DICOM slice - EXACT MATCH for dataset.py
        
        Args:
            pixel_array: Raw DICOM pixel array
            rescale_slope: DICOM RescaleSlope
            rescale_intercept: DICOM RescaleIntercept
            
        Returns:
            tensor: Preprocessed tensor ready for model
        """
        # Apply rescale slope and intercept
        image = pixel_array.astype(float)
        image = image * rescale_slope + rescale_intercept
        
        # Apply windowing (soft tissue window)
        image = self.apply_windowing(image, self.window_center, self.window_width)
        
        # Convert to 8-bit
        image = (image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image).convert('L')
        
        # Resize to 224x224
        pil_image = pil_image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to numpy array
        image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Normalize: (image - mean) / std
        image = (image - self.normalize_mean) / self.normalize_std
        
        # Convert to tensor: (1, H, W)
        tensor = torch.FloatTensor(image).unsqueeze(0)
        
        # Add batch dimension: (1, 1, H, W)
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def detect_orientation(self, dicom_path):
        """
        Detect orientation from DICOM folder
        
        Args:
            dicom_path: Path to DICOM folder
            
        Returns:
            dict: {'orientation': str, 'confidence': float, 'probabilities': dict}
        """
        if self.model is None:
            return {
                'orientation': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'error': 'Model not loaded'
            }
        
        print("\n" + "="*60)
        print("🧭 AI ORIENTATION DETECTION (ResNet18)")
        print("="*60)
        print(f"Input: {dicom_path}")
        
        try:
            # Load DICOM files
            dicom_files = self._load_dicom_files(dicom_path)
            
            if not dicom_files:
                return {
                    'orientation': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': 'No DICOM files found'
                }
            
            # Use MIDDLE slice (most representative)
            mid_idx = len(dicom_files) // 2
            ds = dicom_files[mid_idx]
            
            print(f"✓ Using slice {mid_idx+1}/{len(dicom_files)} (middle slice)")
            
            # Get rescale parameters
            rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
            rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
            
            # Preprocess slice
            image_tensor = self.preprocess_dicom_slice(
                ds.pixel_array, 
                rescale_slope, 
                rescale_intercept
            )
            
            # Predict
            predicted_class, confidence, probabilities = self._predict(image_tensor)
            orientation = self.orientations[predicted_class]
            
            # Create probability dict
            prob_dict = {
                self.orientations[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            # Store results
            self.last_result = {
                'orientation': orientation,
                'confidence': float(confidence),
                'probabilities': prob_dict
            }
            
            # Print results
            self._print_results(orientation, confidence, prob_dict)
            
            return self.last_result
            
        except Exception as e:
            print(f"\n❌ Error during orientation detection: {e}")
            import traceback
            traceback.print_exc()
            return {
                'orientation': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }
    
    def _load_dicom_files(self, dicom_path):
        """Load and sort DICOM files"""
        dicom_folder = Path(dicom_path)
        
        if not dicom_folder.is_dir():
            print(f"❌ Not a directory: {dicom_path}")
            return []
        
        # Find DICOM files
        dicom_files = []
        for file in sorted(dicom_folder.iterdir()):
            if file.suffix.lower() in ['.dcm', ''] and file.is_file():
                try:
                    ds = pydicom.dcmread(str(file), force=True)
                    if hasattr(ds, 'pixel_array'):
                        dicom_files.append(ds)
                except:
                    pass
        
        if not dicom_files:
            print(f"❌ No DICOM files found in {dicom_path}")
            return []
        
        # Sort by slice position
        try:
            dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            try:
                dicom_files.sort(key=lambda x: int(x.InstanceNumber))
            except:
                pass
        
        print(f"✓ Loaded {len(dicom_files)} DICOM slices")
        
        return dicom_files
    
    def _predict(self, image_tensor):
        """Predict orientation"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()
    
    def _print_results(self, orientation, confidence, probabilities):
        """Print detection results"""
        print("\n" + "="*60)
        print("🎯 ORIENTATION DETECTION RESULTS")
        print("="*60)
        print(f"Detected Orientation: {orientation.upper()}")
        print(f"Confidence: {confidence*100:.1f}%")
        print("\n📊 All Probabilities:")
        for name, prob in probabilities.items():
            marker = "◄◄◄" if name == orientation else "   "
            print(f"  {name.capitalize():12s}: {prob*100:6.2f}% {marker}")
        print("="*60)


# Test function
if __name__ == "__main__":
    detector = OrientationDetector(model_path='best_orientation_classifier.pth')
    
    test_dicom = r"C:\path\to\your\dicom\folder"  # CHANGE THIS
    
    if os.path.exists(test_dicom):
        print("\n🚀 Testing orientation detection...\n")
        
        result = detector.detect_orientation(test_dicom)
        
        if 'error' not in result:
            print("\n✅ Success!")
            print(f"Detected: {result['orientation'].upper()}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
        else:
            print(f"\n❌ Error: {result['error']}")
    else:
        print(f"❌ Path not found: {test_dicom}")