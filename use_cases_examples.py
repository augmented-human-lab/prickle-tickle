"""
Use Case Examples for Food Detection and General Object Detection Functions

This file demonstrates various use cases for:
1. detect_all_objects() - General object detection (all objects)
2. detect_food_objects() - Food-specific detection with healthy/unhealthy classification
"""

import cv2
from food_detector import detect_all_objects, detect_food_objects


# ============================================================================
# USE CASE 1: General Object Detection - Detect ALL objects in an image
# ============================================================================

def example_general_detection():
    """Example: Detect all objects in an image (people, cars, food, etc.)"""
    print("=" * 60)
    print("USE CASE 1: General Object Detection")
    print("=" * 60)
    
    # Detect all objects
    detections = detect_all_objects('your_image.jpg', confidence_threshold=0.5)
    
    print(f"\nFound {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['boundary']
        print(f"{i}. {det['label']}")
        print(f"   Boundary: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"   Confidence: {det['confidence']:.2%}\n")
    
    return detections


def example_general_detection_with_filtering():
    """Example: Detect all objects and filter by specific categories"""
    print("=" * 60)
    print("USE CASE 2: General Detection with Category Filtering")
    print("=" * 60)
    
    # Detect all objects
    all_detections = detect_all_objects('your_image.jpg')
    
    # Filter for specific categories
    vehicles = [d for d in all_detections if d['label'] in ['car', 'truck', 'bus', 'motorcycle']]
    people = [d for d in all_detections if d['label'] in ['person']]
    animals = [d for d in all_detections if d['label'] in ['cat', 'dog', 'bird', 'horse']]
    
    print(f"\nVehicles found: {len(vehicles)}")
    print(f"People found: {len(people)}")
    print(f"Animals found: {len(animals)}")
    
    return vehicles, people, animals


def example_general_detection_visualization():
    """Example: Visualize all detected objects with bounding boxes"""
    print("=" * 60)
    print("USE CASE 3: Visualize All Detections")
    print("=" * 60)
    
    # Load image
    img = cv2.imread('your_image.jpg')
    if img is None:
        print("Error: Could not load image")
        return
    
    # Detect all objects
    detections = detect_all_objects(img)
    
    # Draw all bounding boxes
    for det in detections:
        x1, y1, x2, y2 = det['boundary']
        label = f"{det['label']} {det['confidence']:.2f}"
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display result
    cv2.imshow('All Detections', img)
    print(f"Displaying {len(detections)} detected objects. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================================
# USE CASE 2: Food Detection - Detect and classify food items
# ============================================================================

def example_food_detection_all():
    """Example: Detect all objects with food classification (includes non-food)"""
    print("=" * 60)
    print("USE CASE 4: Food Detection - All Objects with Classification")
    print("=" * 60)
    
    # Get all objects with food classification
    detections = detect_food_objects('your_image.jpg', filter_food_only=False)
    
    # Group by classification
    healthy = [d for d in detections if d['classification'] == 'healthy']
    unhealthy = [d for d in detections if d['classification'] == 'unhealthy']
    unknown = [d for d in detections if d['classification'] == 'unknown']
    
    print(f"\nHealthy foods: {len(healthy)}")
    for det in healthy:
        print(f"  - {det['label']} (confidence: {det['confidence']:.2%})")
    
    print(f"\nUnhealthy foods: {len(unhealthy)}")
    for det in unhealthy:
        print(f"  - {det['label']} (confidence: {det['confidence']:.2%})")
    
    print(f"\nOther objects (non-food): {len(unknown)}")
    for det in unknown:
        print(f"  - {det['label']} (confidence: {det['confidence']:.2%})")
    
    return healthy, unhealthy, unknown


def example_food_detection_filtered():
    """Example: Detect ONLY food items (filter out non-food objects)"""
    print("=" * 60)
    print("USE CASE 5: Food Detection - Food Items Only")
    print("=" * 60)
    
    # Get only food items
    food_detections = detect_food_objects('your_image.jpg', filter_food_only=True)
    
    print(f"\nFound {len(food_detections)} food items:")
    for det in food_detections:
        x1, y1, x2, y2 = det['boundary']
        status = "✅ Healthy" if det['classification'] == 'healthy' else "❌ Unhealthy"
        print(f"{status}: {det['label']}")
        print(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Confidence: {det['confidence']:.2%}\n")
    
    return food_detections


def example_food_detection_visualization():
    """Example: Visualize food detections with color coding"""
    print("=" * 60)
    print("USE CASE 6: Visualize Food Detections")
    print("=" * 60)
    
    # Load image
    img = cv2.imread('your_image.jpg')
    if img is None:
        print("Error: Could not load image")
        return
    
    # Detect food items
    detections = detect_food_objects(img, filter_food_only=True)
    
    # Draw with color coding
    for det in detections:
        x1, y1, x2, y2 = det['boundary']
        
        # Color code: Green for healthy, Red for unhealthy
        if det['classification'] == 'healthy':
            color = (0, 255, 0)  # Green
            label = f"{det['label']} (Healthy)"
        else:
            color = (0, 0, 255)  # Red
            label = f"{det['label']} (Unhealthy)"
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Draw label
        cv2.putText(img, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add summary text
    healthy_count = sum(1 for d in detections if d['classification'] == 'healthy')
    unhealthy_count = sum(1 for d in detections if d['classification'] == 'unhealthy')
    summary = f"Healthy: {healthy_count} | Unhealthy: {unhealthy_count}"
    cv2.putText(img, summary, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display
    cv2.imshow('Food Detection', img)
    print(f"Displaying {len(detections)} food items. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================================
# USE CASE 3: Combined Use Cases
# ============================================================================

def example_combined_analysis():
    """Example: Use both functions for comprehensive image analysis"""
    print("=" * 60)
    print("USE CASE 7: Combined Analysis - General + Food Detection")
    print("=" * 60)
    
    image_path = 'your_image.jpg'
    
    # Get all objects
    all_objects = detect_all_objects(image_path)
    
    # Get food-specific information
    food_objects = detect_food_objects(image_path, filter_food_only=True)
    
    # Analysis
    print(f"\n📊 Image Analysis Summary:")
    print(f"  Total objects detected: {len(all_objects)}")
    print(f"  Food items found: {len(food_objects)}")
    print(f"  Non-food objects: {len(all_objects) - len(food_objects)}")
    
    if food_objects:
        healthy = [f for f in food_objects if f['classification'] == 'healthy']
        unhealthy = [f for f in food_objects if f['classification'] == 'unhealthy']
        
        print(f"\n🍎 Food Breakdown:")
        print(f"  Healthy foods: {len(healthy)}")
        print(f"  Unhealthy foods: {len(unhealthy)}")
        
        if len(healthy) > len(unhealthy):
            print("  ✅ This meal is mostly healthy!")
        elif len(unhealthy) > len(healthy):
            print("  ⚠️  This meal contains mostly unhealthy items")
        else:
            print("  ⚖️  Mixed meal")
    
    return all_objects, food_objects


def example_real_time_webcam():
    """Example: Real-time detection using webcam (general objects)"""
    print("=" * 60)
    print("USE CASE 8: Real-time General Object Detection")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit, 'f' to switch to food-only mode")
    food_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if food_mode:
            # Food detection mode
            detections = detect_food_objects(frame, filter_food_only=True)
            for det in detections:
                x1, y1, x2, y2 = det['boundary']
                color = (0, 255, 0) if det['classification'] == 'healthy' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{det['label']} ({det['classification']})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            mode_text = "Food Detection Mode"
        else:
            # General detection mode
            detections = detect_all_objects(frame)
            for det in detections:
                x1, y1, x2, y2 = det['boundary']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            mode_text = "General Detection Mode"
        
        # Display mode and count
        cv2.putText(frame, f"{mode_text} - Objects: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Object Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            food_mode = not food_mode
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# Main execution examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FOOD DETECTION AND GENERAL OBJECT DETECTION - USE CASES")
    print("=" * 60)
    print("\nNote: Replace 'your_image.jpg' with your actual image path")
    print("\nAvailable examples:")
    print("1. example_general_detection() - Basic general detection")
    print("2. example_general_detection_with_filtering() - Filter by category")
    print("3. example_general_detection_visualization() - Visualize all objects")
    print("4. example_food_detection_all() - All objects with food classification")
    print("5. example_food_detection_filtered() - Food items only")
    print("6. example_food_detection_visualization() - Visualize food with colors")
    print("7. example_combined_analysis() - Combined analysis")
    print("8. example_real_time_webcam() - Real-time webcam detection")
    print("\n" + "=" * 60)
    
    # Uncomment the example you want to run:
    
    # Example 1: General detection
    # example_general_detection()
    
    # Example 2: Food detection (all objects)
    # example_food_detection_all()
    
    # Example 3: Food detection (food only)
    # example_food_detection_filtered()
    
    # Example 4: Combined analysis
    # example_combined_analysis()

