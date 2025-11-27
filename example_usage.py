"""
Quick examples of detect_all_objects and detect_food_objects functions
For more comprehensive examples, see use_cases_examples.py
"""

from food_detector import detect_all_objects, detect_food_objects
import cv2

# ============================================================================
# GENERAL OBJECT DETECTION - detect_all_objects()
# ============================================================================

print("=" * 60)
print("GENERAL OBJECT DETECTION - detect_all_objects()")
print("=" * 60)

# Example 1: Detect ALL objects in an image
print("\nExample 1: Detect all objects")
all_detections = detect_all_objects('your_image.jpg', confidence_threshold=0.5)

print(f"Found {len(all_detections)} objects:")
for det in all_detections:
    x1, y1, x2, y2 = det['boundary']
    print(f"  - {det['label']} at ({x1}, {y1}) to ({x2}, {y2}) - confidence: {det['confidence']:.2f}")

# Example 2: Using with numpy array
print("\nExample 2: Using with numpy array")
img = cv2.imread('your_image.jpg')
if img is not None:
    detections = detect_all_objects(img)
    
    # Draw all bounding boxes
    for det in detections:
        x1, y1, x2, y2 = det['boundary']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{det['label']} {det['confidence']:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('All Detections', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================================
# FOOD DETECTION - detect_food_objects()
# ============================================================================

print("\n" + "=" * 60)
print("FOOD DETECTION - detect_food_objects()")
print("=" * 60)

# Example 3: Detect all objects with food classification
print("\nExample 3: All objects with food classification")
all_with_classification = detect_food_objects('your_image.jpg', filter_food_only=False)

for det in all_with_classification:
    x1, y1, x2, y2 = det['boundary']
    print(f"  - {det['label']}: {det['classification']} (confidence: {det['confidence']:.2f})")

# Example 4: Detect ONLY food items (filter out non-food)
print("\nExample 4: Food items only")
food_only = detect_food_objects('your_image.jpg', filter_food_only=True)

print(f"Found {len(food_only)} food items:")
for det in food_only:
    x1, y1, x2, y2 = det['boundary']
    status = "✅ Healthy" if det['classification'] == 'healthy' else "❌ Unhealthy"
    print(f"  {status}: {det['label']} at ({x1}, {y1}) to ({x2}, {y2})")

# Example 5: Filter by classification
print("\nExample 5: Filter by classification")
all_detections = detect_food_objects('your_image.jpg', filter_food_only=False)

healthy_foods = [d for d in all_detections if d['classification'] == 'healthy']
unhealthy_foods = [d for d in all_detections if d['classification'] == 'unhealthy']
non_food = [d for d in all_detections if d['classification'] == 'unknown']

print(f"Healthy foods: {len(healthy_foods)}")
print(f"Unhealthy foods: {len(unhealthy_foods)}")
print(f"Non-food objects: {len(non_food)}")

