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

### Detect food in an image:
```bash
python food_detector.py path/to/your/image.jpg
```

### Use webcam for real-time detection:
```bash
python food_detector.py --webcam
# or
python food_detector.py -w
```

Press 'q' to quit when using webcam.

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

