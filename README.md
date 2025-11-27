# Food Detection - Healthy/Unhealthy Classifier

A simple Python script to detect and classify food items as healthy or unhealthy using YOLOv8 object detection.

## Features

- 🍎 Detects food items in images or webcam feed
- ✅ Classifies foods as healthy or unhealthy
- 🚀 Fast and lightweight (uses YOLOv8 nano model)
- 💻 Runs efficiently on laptop CPU

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

The script will automatically download the YOLOv8 nano model on first run.

## Usage

### Command Line Usage

#### Detect food in an image:
```bash
python food_detector.py path/to/your/image.jpg
```

#### Use webcam for real-time detection:
```bash
python food_detector.py --webcam
# or
python food_detector.py -w
```

Press 'q' to quit when using webcam.

### Python API - Integration Functions

The module provides two main functions for integration with other scripts:

#### 1. `detect_all_objects()` - General Object Detection

Detects **ALL objects** in an image (people, cars, food, animals, etc.) without food classification.

```python
from food_detector import detect_all_objects

# Detect all objects
detections = detect_all_objects('image.jpg')

# Each detection contains:
# - 'boundary': (x1, y1, x2, y2) - bounding box coordinates
# - 'label': object name (e.g., 'person', 'car', 'apple')
# - 'confidence': confidence score (0.0-1.0)

for det in detections:
    print(f"{det['label']} at {det['boundary']} - confidence: {det['confidence']:.2f}")
```

#### 2. `detect_food_objects()` - Food Detection with Classification

Detects objects and classifies them as healthy/unhealthy food.

```python
from food_detector import detect_food_objects

# Option 1: Get all objects with food classification (includes non-food as 'unknown')
all_detections = detect_food_objects('image.jpg', filter_food_only=False)

# Option 2: Get ONLY food items (filters out non-food objects)
food_only = detect_food_objects('image.jpg', filter_food_only=True)

# Each detection contains:
# - 'boundary': (x1, y1, x2, y2) - bounding box coordinates
# - 'label': object name
# - 'confidence': confidence score (0.0-1.0)
# - 'classification': 'healthy', 'unhealthy', or 'unknown'

for det in food_only:
    print(f"{det['label']}: {det['classification']}")
```

**Parameters:**
- `image`: Image path (str) or numpy array (BGR format)
- `confidence_threshold`: Minimum confidence score (default: 0.5)
- `filter_food_only`: If True, only return food items (default: False)
- `model`: Optional YOLO model instance (auto-loaded if None)

### Example Files

- `example_usage.py` - Quick examples of both functions
- `use_cases_examples.py` - Comprehensive use case examples with 8 different scenarios

## How it works

1. Uses YOLOv8 nano model for object detection (lightweight, fast)
2. Detects food items in the image/video
3. Classifies detected items as healthy or unhealthy based on predefined categories
4. Displays results with color-coded bounding boxes:
   - 🟢 Green = Healthy food
   - 🔴 Red = Unhealthy food
   - ⚪ Gray = Unknown/not classified

## Customization

You can extend the `FOOD_CLASSIFICATION` dictionary in `food_detector.py` to add more food categories and their classifications.

## Notes

- The YOLOv8 COCO model has limited food classes. For better food detection, you may want to fine-tune the model on a food-specific dataset.
- The classification is rule-based. For more accurate classification, consider training a separate classifier model.

