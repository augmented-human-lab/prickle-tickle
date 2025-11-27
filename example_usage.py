"""
Example usage of detect_food_objects function for integration
"""

from food_detector import detect_food_objects

# Example 1: Detect from image file
print("Example 1: Detecting from image file")
detections = detect_food_objects('your_image.jpg', confidence_threshold=0.5)

for det in detections:
    x1, y1, x2, y2 = det['boundary']
    print(f"Found: {det['label']}")
    print(f"  Boundary: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Confidence: {det['confidence']:.2f}")
    print(f"  Classification: {det['classification']}")
    print()

# Example 2: Using with numpy array (from OpenCV)
import cv2

print("Example 2: Detecting from numpy array")
img = cv2.imread('your_image.jpg')
if img is not None:
    detections = detect_food_objects(img, confidence_threshold=0.5)
    
    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = det['boundary']
        color = (0, 255, 0) if det['classification'] == 'healthy' else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{det['label']} ({det['classification']})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Detections', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example 3: Filter only healthy/unhealthy foods
print("Example 3: Filtering by classification")
all_detections = detect_food_objects('your_image.jpg')

healthy_foods = [d for d in all_detections if d['classification'] == 'healthy']
unhealthy_foods = [d for d in all_detections if d['classification'] == 'unhealthy']

print(f"Healthy foods found: {len(healthy_foods)}")
print(f"Unhealthy foods found: {len(unhealthy_foods)}")

