#!/usr/bin/env python3
"""
Demographic Classification Module for Bias Analysis

This module implements demographic classification using face detection and
demographic analysis to detect bias in generated images.
"""

import os
# Set environment variable to fix Keras 3 compatibility with DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import deepface for better demographic classification
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace available - will use for demographic classification")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available. Install with: pip install deepface")
    logger.warning("Will use fallback face detection method.")

def convert_to_native_types(obj):
    """Convert numpy types and other non-JSON-serializable types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

class FairFaceClassifier:
    """FairFace demographic classifier implementation."""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Initialize the FairFace classifier.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.race_labels = ['White', 'Black', 'Indian', 'East Asian', 'Southeast Asian', 'Middle Eastern', 'Latino']
        self.gender_labels = ['Male', 'Female']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        
        self._load_model(model_path)
        self._setup_transforms()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self, model_path: str):
        """Load the demographic classification model."""
        if DEEPFACE_AVAILABLE:
            logger.info("Using DeepFace for demographic classification")
            self.model = None  # DeepFace handles model loading internally
            return
        
        # Fallback to custom model
        try:
            logger.info("Using custom ResNet-based classifier")
            self.model = self._create_fairface_model()
            
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'])
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning("No pre-trained model found. Attempting to use face detection + heuristics.")
                # Don't use random weights - use face detection instead
                self.model = None
            
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _create_fairface_model(self):
        """Create the FairFace model architecture."""
        # Simplified FairFace model for demo
        class FairFaceModel(nn.Module):
            def __init__(self, num_races=7, num_genders=2, num_ages=9):
                super().__init__()
                self.backbone = resnet34(pretrained=True)
                self.backbone.fc = nn.Identity()  # Remove final classification layer
                
                # Task-specific heads
                self.race_head = nn.Linear(512, num_races)
                self.gender_head = nn.Linear(512, num_genders)
                self.age_head = nn.Linear(512, num_ages)
            
            def forward(self, x):
                features = self.backbone(x)
                race_logits = self.race_head(features)
                gender_logits = self.gender_head(features)
                age_logits = self.age_head(features)
                return race_logits, gender_logits, age_logits
        
        return FairFaceModel()
    
    def _create_fallback_model(self):
        """Create a fallback model when FairFace is not available."""
        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = resnet34(pretrained=True)
                self.backbone.fc = nn.Identity()
                
                # Simple classification heads
                self.race_head = nn.Linear(512, 7)
                self.gender_head = nn.Linear(512, 2)
                self.age_head = nn.Linear(512, 9)
            
            def forward(self, x):
                features = self.backbone(x)
                race_logits = self.race_head(features)
                gender_logits = self.gender_head(features)
                age_logits = self.age_head(features)
                return race_logits, gender_logits, age_logits
        
        return FallbackModel()
    
    def _setup_transforms(self):
        """Set up image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for classification."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify demographic attributes of an image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary containing race, gender, age predictions and confidence scores
        """
        # Convert PIL to numpy array for OpenCV
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Try DeepFace first (most accurate)
        if DEEPFACE_AVAILABLE:
            try:
                return self._classify_with_deepface(img_cv)
            except Exception as e:
                logger.warning(f"DeepFace classification failed: {e}. Trying fallback method.")
        
        # Fallback: Use face detection + heuristics
        return self._classify_with_face_detection(img_cv)
    
    def _classify_with_deepface(self, img_cv) -> Dict[str, Any]:
        """Classify using DeepFace library."""
        try:
            # Save image temporarily for DeepFace (it requires file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, img_cv)
                temp_path = tmp_file.name
            
            try:
                # DeepFace analyzes the image
                # Use opencv as default as it's most stable and doesn't use TF
                backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
                backend = 'opencv'
                
                # Try to find a better backend if available and safe
                # But given the Keras errors, we'll stick to opencv or ssd if possible
                # unless user explicitly wants others. For now, let's try opencv first.
                
                try:
                    result = DeepFace.analyze(
                        img_path=temp_path,
                        actions=['age', 'gender', 'race'],
                        detector_backend=backend,
                        enforce_detection=False,
                        silent=True
                    )
                except Exception as e:
                    # If opencv fails, try ssd
                    logger.warning(f"DeepFace with opencv failed: {e}. Trying ssd.")
                    try:
                        result = DeepFace.analyze(
                            img_path=temp_path,
                            actions=['age', 'gender', 'race'],
                            detector_backend='ssd',
                            enforce_detection=False,
                            silent=True
                        )
                    except Exception as e2:
                        logger.error(f"DeepFace with ssd failed: {e2}")
                        raise e
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # Handle both single dict and list responses
            if isinstance(result, list):
                # Find the largest face if multiple are detected
                if len(result) > 1:
                    # DeepFace result usually contains 'region' key with {'x', 'y', 'w', 'h'}
                    try:
                        result = max(result, key=lambda x: x.get('region', {}).get('w', 0) * x.get('region', {}).get('h', 0))
                    except:
                        result = result[0]
                else:
                    result = result[0]
            
            # Map DeepFace outputs to our labels
            race_mapping = {
                'white': 'White',
                'black': 'Black',
                'asian': 'East Asian',
                'indian': 'Indian',
                'middle eastern': 'Middle Eastern',
                'latino hispanic': 'Latino',
                'hispanic': 'Latino',
                'southeast asian': 'Southeast Asian'
            }
            
            gender_mapping = {
                'Man': 'Male',
                'Woman': 'Female',
                'male': 'Male',
                'female': 'Female'
            }
            
            # Extract race (dominant race)
            race_dict = result.get('race', {})
            if race_dict:
                dominant_race = max(race_dict.items(), key=lambda x: x[1])
                race_name = race_mapping.get(dominant_race[0].lower(), dominant_race[0])
                race_confidence = dominant_race[1] / 100.0
            else:
                race_name = 'Unknown'
                race_confidence = 0.0
            
            # Extract gender
            gender_dict = result.get('gender', {})
            if gender_dict:
                dominant_gender = max(gender_dict.items(), key=lambda x: x[1])
                gender_name = gender_mapping.get(dominant_gender[0], dominant_gender[0])
                gender_confidence = dominant_gender[1] / 100.0
            else:
                gender_name = 'Unknown'
                gender_confidence = 0.0
            
            # Extract age
            age = result.get('age', 0)
            age_confidence = 0.8  # DeepFace doesn't provide age confidence
            
            # Map age to age group
            if age < 3:
                age_group = '0-2'
            elif age < 10:
                age_group = '3-9'
            elif age < 20:
                age_group = '10-19'
            elif age < 30:
                age_group = '20-29'
            elif age < 40:
                age_group = '30-39'
            elif age < 50:
                age_group = '40-49'
            elif age < 60:
                age_group = '50-59'
            elif age < 70:
                age_group = '60-69'
            else:
                age_group = '70+'
            
            return {
                'race': {
                    'prediction': race_name,
                    'confidence': race_confidence,
                    'all_scores': {}
                },
                'gender': {
                    'prediction': gender_name,
                    'confidence': gender_confidence,
                    'all_scores': {}
                },
                'age': {
                    'prediction': age_group,
                    'confidence': age_confidence,
                    'all_scores': {}
                }
            }
        except Exception as e:
            logger.error(f"DeepFace classification error: {e}")
            raise
    
    def _classify_with_face_detection(self, img_cv) -> Dict[str, Any]:
        """Fallback classification using face detection and heuristics."""
        # Load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # No face detected - return unknown
            logger.warning("No face detected in image")
            return {
                'race': {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'all_scores': {}
                },
                'gender': {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'all_scores': {}
                },
                'age': {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'all_scores': {}
                }
            }
        
        # Use the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = img_cv[y:y+h, x:x+w]
        
        # If we have a model, use it
        if self.model is not None:
            try:
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                input_tensor = self.preprocess_image(face_pil).to(self.device)
                
                with torch.no_grad():
                    race_logits, gender_logits, age_logits = self.model(input_tensor)
                    
                    race_probs = torch.softmax(race_logits, dim=1)
                    gender_probs = torch.softmax(gender_logits, dim=1)
                    age_probs = torch.softmax(age_logits, dim=1)
                    
                    race_pred = torch.argmax(race_probs, dim=1).item()
                    gender_pred = torch.argmax(gender_probs, dim=1).item()
                    age_pred = torch.argmax(age_probs, dim=1).item()
                    
                    return {
                        'race': {
                            'prediction': self.race_labels[race_pred],
                            'confidence': race_probs[0, race_pred].item(),
                            'all_scores': race_probs[0].cpu().numpy().tolist()
                        },
                        'gender': {
                            'prediction': self.gender_labels[gender_pred],
                            'confidence': gender_probs[0, gender_pred].item(),
                            'all_scores': gender_probs[0].cpu().numpy().tolist()
                        },
                        'age': {
                            'prediction': self.age_labels[age_pred],
                            'confidence': age_probs[0, age_pred].item(),
                            'all_scores': age_probs[0].cpu().numpy().tolist()
                        }
                    }
            except Exception as e:
                logger.error(f"Model classification failed: {e}")
        
        # Final fallback: return unknown (face detected but can't classify)
        logger.warning("Face detected but classification failed")
        return {
            'race': {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'all_scores': {}
            },
            'gender': {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'all_scores': {}
            },
            'age': {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
        }
    
    def classify_batch(self, image_paths: List[str], output_path: str = None) -> List[Dict[str, Any]]:
        """
        Classify a batch of images.
        
        Args:
            image_paths: List of paths to images
            output_path: Optional path to save results
            
        Returns:
            List of classification results
        """
        results = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Classifying image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Load image
                image = Image.open(image_path)
                
                # Classify
                classification = self.classify_image(image)
                classification['image_path'] = image_path
                classification['image_name'] = os.path.basename(image_path)
                
                # Check if classification was successful
                if classification.get('race', {}).get('prediction') == 'Unknown':
                    failed += 1
                    logger.warning(f"Could not classify demographics for {os.path.basename(image_path)}")
                else:
                    successful += 1
                
                results.append(classification)
                
            except Exception as e:
                failed += 1
                logger.error(f"Error classifying {image_path}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # Add error result
                results.append({
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'error': str(e),
                    'race': {'prediction': 'Unknown', 'confidence': 0.0},
                    'gender': {'prediction': 'Unknown', 'confidence': 0.0},
                    'age': {'prediction': 'Unknown', 'confidence': 0.0}
                })
        
        logger.info(f"Classification complete: {successful} successful, {failed} failed out of {len(image_paths)} images")
        
        # Save results if output path provided
        if output_path:
            # Convert numpy types to native Python types for JSON serialization
            results_serializable = convert_to_native_types(results)
            with open(output_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            logger.info(f"Classification results saved to {output_path}")
        
        return results

class BiasAnalyzer:
    """Analyze demographic bias in classification results."""
    
    def __init__(self, race_labels: List[str], gender_labels: List[str]):
        self.race_labels = race_labels
        self.gender_labels = gender_labels
    
    def compute_statistical_parity_difference(self, results: List[Dict[str, Any]], 
                                            attribute: str, group1: str, group2: str) -> float:
        """
        Compute Statistical Parity Difference (SPD).
        
        SPD = P(Y=1|A=group1) - P(Y=1|A=group2)
        """
        group1_count = sum(1 for r in results if r.get(attribute, {}).get('prediction') == group1)
        group2_count = sum(1 for r in results if r.get(attribute, {}).get('prediction') == group2)
        
        total = len(results)
        if total == 0:
            return 0.0
        
        group1_rate = group1_count / total
        group2_rate = group2_count / total
        
        return group1_rate - group2_rate
    
    def compute_representation_ratio(self, results: List[Dict[str, Any]], 
                                   attribute: str, group: str) -> float:
        """Compute representation ratio for a demographic group."""
        group_count = sum(1 for r in results if r.get(attribute, {}).get('prediction') == group)
        total = len(results)
        
        if total == 0:
            return 0.0
        
        return group_count / total
    
    def analyze_bias(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive bias analysis."""
        analysis = {
            'race_distribution': {},
            'gender_distribution': {},
            'statistical_parity_differences': {},
            'representation_ratios': {}
        }
        
        # Race distribution
        for race in self.race_labels:
            analysis['race_distribution'][race] = self.compute_representation_ratio(
                results, 'race', race
            )
        
        # Gender distribution
        for gender in self.gender_labels:
            analysis['gender_distribution'][gender] = self.compute_representation_ratio(
                results, 'gender', gender
            )
        
        # Statistical Parity Differences for race
        for i, race1 in enumerate(self.race_labels):
            for race2 in self.race_labels[i+1:]:
                spd = self.compute_statistical_parity_difference(results, 'race', race1, race2)
                analysis['statistical_parity_differences'][f'{race1}_vs_{race2}'] = spd
        
        # Statistical Parity Differences for gender
        spd_gender = self.compute_statistical_parity_difference(
            results, 'gender', 'Male', 'Female'
        )
        analysis['statistical_parity_differences']['Male_vs_Female'] = spd_gender
        
        return analysis

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Classify demographics in generated images")
    parser.add_argument("--input", required=True, 
                       help="Directory containing images to classify")
    parser.add_argument("--output", required=True,
                       help="Output file for classification results")
    parser.add_argument("--model-path", 
                       help="Path to pre-trained FairFace model")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(args.input).glob(f"*{ext}"))
        image_paths.extend(Path(args.input).glob(f"*{ext.upper()}"))
    
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        logger.error(f"No images found in {args.input}")
        return
    
    logger.info(f"Found {len(image_paths)} images to classify")
    
    # Initialize classifier
    classifier = FairFaceClassifier(model_path=args.model_path, device=args.device)
    
    # Classify images
    results = classifier.classify_batch(image_paths, args.output)
    
    # Analyze bias
    analyzer = BiasAnalyzer(classifier.race_labels, classifier.gender_labels)
    bias_analysis = analyzer.analyze_bias(results)
    
    # Save bias analysis
    bias_output = args.output.replace('.json', '_bias_analysis.json')
    # Convert numpy types to native Python types for JSON serialization
    bias_analysis_serializable = convert_to_native_types(bias_analysis)
    with open(bias_output, 'w') as f:
        json.dump(bias_analysis_serializable, f, indent=2)
    
    logger.info(f"Classification complete! Results saved to {args.output}")
    logger.info(f"Bias analysis saved to {bias_output}")

if __name__ == "__main__":
    main()