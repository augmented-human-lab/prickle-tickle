"""
Simple Food Detection Script - Healthy/Unhealthy Classification
Uses YOLOv8 for object detection and classifies foods as healthy/unhealthy
"""

import cv2
from ultralytics import YOLO
import sys
from typing import List, Dict, Union, Tuple
import numpy as np

# Define healthy and unhealthy food categories
# Based on COCO dataset classes that YOLOv8 can detect
HEALTHY_FOODS = {
    'apple', 'banana', 'orange', 'broccoli', 'carrot', 
    'salad', 'sandwich'  # Note: YOLOv8 COCO model has limited food classes
}

UNHEALTHY_FOODS = {
    'pizza', 'hot dog', 'donut', 'cake'
}

# Extended mapping for common food items (you can expand this)
FOOD_CLASSIFICATION = {
    # Healthy
    'apple': 'healthy',
    'banana': 'healthy',
    'orange': 'healthy',
    'broccoli': 'healthy',
    'carrot': 'healthy',
    'sandwich': 'healthy', 
    
    # Unhealthy
    'pizza': 'unhealthy',
    'hot dog': 'unhealthy',
    'donut': 'unhealthy',
    'cake': 'unhealthy',
    'sandwich': 'unhealthy', 
    'laptop': 'unhealthy',  # Non-food item for testing
    'cup': 'unhealthy'  # Non-food item for testing
}


def classify_food(class_name):
    """Classify a detected food item as healthy or unhealthy"""
    class_name_lower = class_name.lower()
    
    # Check direct mapping
    if class_name_lower in FOOD_CLASSIFICATION:
        return FOOD_CLASSIFICATION[class_name_lower]
    
    # Check healthy foods set
    if class_name_lower in HEALTHY_FOODS:
        return 'healthy'
    
    # Check unhealthy foods set
    if class_name_lower in UNHEALTHY_FOODS:
        return 'unhealthy'
    
    return 'unknown'


# Global model instance (loaded on first use)
_model = None


def get_model():
    """Get or load the YOLOv8 model (singleton pattern)"""
    global _model
    if _model is None:
        _model = YOLO('yolov8n.pt')  # nano version - fastest, good for laptops
    return _model


def detect_all_objects(image: Union[str, np.ndarray], 
                       confidence_threshold: float = 0.5,
                       model: YOLO = None) -> List[Dict]:
    """
    Detect ALL objects in an image and return boundaries and labels.
    This is a general object detection function that returns all detected objects
    without any food-specific classification.
    
    Args:
        image: Image path (str) or numpy array (BGR format)
        confidence_threshold: Minimum confidence score (0.0-1.0)
        model: Optional YOLO model instance (if None, will load/create one)
    
    Returns:
        List of dictionaries, each containing:
            - 'boundary': Tuple of (x1, y1, x2, y2) bounding box coordinates
            - 'label': String label of detected object
            - 'confidence': Float confidence score (0.0-1.0)
    
    Example:
        >>> detections = detect_all_objects('image.jpg')
        >>> for det in detections:
        ...     print(f"{det['label']} at {det['boundary']} with confidence {det['confidence']:.2f}")
    """
    # Load model if not provided
    if model is None:
        model = get_model()
    
    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        img = image.copy()
    
    # Run detection
    results = model(img, verbose=False)
    
    # Extract all detections
    detections = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Filter by confidence threshold
            if confidence >= confidence_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boundary = (x1, y1, x2, y2)
                
                detections.append({
                    'boundary': boundary,
                    'label': class_name,
                    'confidence': confidence
                })
    
    return detections


def detect_food_objects(image: Union[str, np.ndarray], 
                        confidence_threshold: float = 0.5,
                        filter_food_only: bool = False,
                        model: YOLO = None) -> List[Dict]:
    """
    Detect food objects in an image and return boundaries and labels with classification.
    
    This function detects objects and classifies them as healthy/unhealthy food.
    By default, it returns all detected objects with their food classification
    (non-food items will have classification='unknown').
    
    Args:
        image: Image path (str) or numpy array (BGR format)
        confidence_threshold: Minimum confidence score (0.0-1.0)
        filter_food_only: If True, only return objects classified as food (healthy/unhealthy).
                         If False, return all objects with classification info.
        model: Optional YOLO model instance (if None, will load/create one)
    
    Returns:
        List of dictionaries, each containing:
            - 'boundary': Tuple of (x1, y1, x2, y2) bounding box coordinates
            - 'label': String label of detected object
            - 'confidence': Float confidence score (0.0-1.0)
            - 'classification': String 'healthy', 'unhealthy', or 'unknown'
    
    Example:
        >>> # Get all objects with food classification
        >>> detections = detect_food_objects('food_image.jpg')
        >>> 
        >>> # Get only food items (filter out non-food)
        >>> food_only = detect_food_objects('food_image.jpg', filter_food_only=True)
    """
    # Load model if not provided
    if model is None:
        model = get_model()
    
    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        img = image.copy()
    
    # Run detection
    results = model(img, verbose=False)
    
    # Extract detections
    detections = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Filter by confidence threshold
            if confidence >= confidence_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boundary = (x1, y1, x2, y2)
                
                # Classify food
                classification = classify_food(class_name)
                
                # Filter out non-food items if requested
                if filter_food_only and classification == 'unknown':
                    continue
                
                detections.append({
                    'boundary': boundary,
                    'label': class_name,
                    'confidence': confidence,
                    'classification': classification
                })
    
    return detections


def detect(image: Union[str, np.ndarray], 
           confidence_threshold: float = 0.5) -> Tuple[Tuple[int, int, int, int], str, str]:
    """
    Simple detection function that returns the first detected food item.
    Returns default values if no food is detected.
    
    Args:
        image: Image path (str) or numpy array (BGR format)
        confidence_threshold: Minimum confidence score (0.0-1.0)
    
    Returns:
        Tuple of (food_box, food_label, food_type):
            - food_box: (xmin, ymin, xmax, ymax) bounding box coordinates
            - food_label: Label of detected object
            - food_type: 'good' or 'bad' classification
    
    Example:
        >>> food_box, food_label, food_type = detect(img)
    """
    # Detect food objects in the image
    detections = detect_food_objects(image, confidence_threshold=confidence_threshold, 
                                     filter_food_only=False)
    
    # If food detected, return the first one
    if detections:
        detection = detections[0]
        food_box = detection['boundary']
        food_label = detection['label']
        
        # Map classification to 'good' or 'bad'
        classification = detection['classification']
        if classification == 'healthy':
            food_type = 'good'
        elif classification == 'unhealthy':
            food_type = 'bad'
        else:
            food_type = 'unknown'
        
        return food_box, food_label, food_type
    
    # Return default values if nothing detected
    # Using center of image as default box
    return (0, 0, 100, 100), "none", "unknown"


def detect_food(image_path=None, use_webcam=False):
    """
    Detect and classify food in an image or webcam feed
    
    Args:
        image_path: Path to image file (optional if use_webcam=True)
        use_webcam: If True, use webcam instead of image file
    """
    # Load YOLOv8 model (nano version for speed on laptop)
    print("Loading YOLOv8 model...")
    model = get_model()
    
    if use_webcam:
        # Use webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            # Process results
            healthy_count = 0
            unhealthy_count = 0
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Only process if confidence is high enough
                    if confidence > 0.5:
                        classification = classify_food(class_name)
                        
                        # Draw bounding box with color coding
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if classification == 'healthy':
                            color = (0, 255, 0)  # Green
                            healthy_count += 1
                            label = f"{class_name} (Healthy) {confidence:.2f}"
                        elif classification == 'unhealthy':
                            color = (0, 0, 255)  # Red
                            unhealthy_count += 1
                            label = f"{class_name} (Unhealthy) {confidence:.2f}"
                        else:
                            color = (128, 128, 128)  # Gray
                            label = f"{class_name} {confidence:.2f}"
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display summary
            summary = f"Healthy: {healthy_count} | Unhealthy: {unhealthy_count}"
            cv2.putText(frame, summary, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Food Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        # Use image file
        if not image_path:
            print("Error: Please provide an image path or use --webcam flag")
            return
        
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Use the main detection function
        detections = detect_food_objects(image_path, confidence_threshold=0.5, model=model)
        
        # Process and display results
        healthy_items = []
        unhealthy_items = []
        
        # Load image for display
        image = cv2.imread(image_path)
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['boundary']
            label = det['label']
            confidence = det['confidence']
            classification = det['classification']
            
            # Draw bounding box
            if classification == 'healthy':
                color = (0, 255, 0)  # Green
                healthy_items.append(f"{label} ({confidence:.2f})")
                display_label = f"{label} (Healthy) {confidence:.2f}"
            elif classification == 'unhealthy':
                color = (0, 0, 255)  # Red
                unhealthy_items.append(f"{label} ({confidence:.2f})")
                display_label = f"{label} (Unhealthy) {confidence:.2f}"
            else:
                color = (128, 128, 128)  # Gray
                display_label = f"{label} {confidence:.2f}"
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, display_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Print results
        print("\n=== Detection Results ===")
        if healthy_items:
            print(f"\nHealthy Foods Detected ({len(healthy_items)}):")
            for item in healthy_items:
                print(f"  - {item}")
        
        if unhealthy_items:
            print(f"\nUnhealthy Foods Detected ({len(unhealthy_items)}):")
            for item in unhealthy_items:
                print(f"  - {item}")
        
        if not healthy_items and not unhealthy_items:
            print("\nNo food items detected (or detected items are not in classification list)")
        
        # Display image
        cv2.imshow('Food Detection', annotated_image)
        print("\nPress any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--webcam" or sys.argv[1] == "-w":
            detect_food(use_webcam=True)
    # Integration function for prickletickle.py
    # Returns (food_box, food_label, food_type) for first detected food
    # food_type is 'good' or 'bad'
        else:
            detect_food(image_path=sys.argv[1])
    else:
        print("Usage:")
        print("  python food_detector.py <image_path>  # Detect in image")
        print("  python food_detector.py --webcam      # Use webcam")
        print("  python food_detector.py -w            # Use webcam (short)")
        print("\nFor integration with other scripts, use detect_food_objects():")
        print("  from food_detector import detect_food_objects")
        print("  detections = detect_food_objects('image.jpg')")
        print("  for det in detections:")
        print("      print(f\"{det['label']} at {det['boundary']} - {det['classification']}\")")
    def detect(img):
        detections = detect_food_objects(img, confidence_threshold=0.5, filter_food_only=True)
        if not detections:
            # No food detected, return dummy values
            return (0, 0, 0, 0), "unknown", "unknown"
        det = detections[0]
        food_box = det['boundary']
        food_label = det['label']
        classification = det['classification']
        # Map to 'good'/'bad'
        if classification == 'healthy':
            food_type = 'good'
        elif classification == 'unhealthy':
            food_type = 'bad'
        else:
            food_type = 'unknown'
        return food_box, food_label, food_type

