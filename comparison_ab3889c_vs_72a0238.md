# Side-by-Side Comparison: ab3889c vs 72a0238

## Commit Information

**Commit 1 (ab3889c):** Your commit
- Author: Yifei.Luo
- Message: "Called food detector in prickletickle. Changed food box to list of dict. Added unhealthy food filter."

**Commit 2 (72a0238):** Merge commit
- Message: "Merge branch 'main' of https://github.com/augmented-human-lab/prickle-tickle"

---

## 1. food_detector.py - FOOD_CLASSIFICATION Dictionary

### Your Version (ab3889c):
```python
FOOD_CLASSIFICATION = {
    # Healthy
    'apple': 'healthy',
    'banana': 'healthy',
    'orange': 'healthy',
    'broccoli': 'healthy',
    'carrot': 'healthy',
    'sandwich': 'healthy',  # Can be healthy depending on ingredients
    
    # Unhealthy
    'pizza': 'unhealthy',
    'hot dog': 'unhealthy',
    'donut': 'unhealthy',
    'cake': 'unhealthy',
    'bottle': 'unhealthy',  # ✅ Your addition
}
```

### Merge Version (72a0238):
```python
FOOD_CLASSIFICATION = {
    # Healthy
    'apple': 'healthy',
    'banana': 'healthy',
    'orange': 'healthy',
    'broccoli': 'healthy',
    'carrot': 'healthy',
    'sandwich': 'healthy',  # ⚠️ Comment removed
    
    # Unhealthy
    'pizza': 'unhealthy',
    'hot dog': 'unhealthy',
    'donut': 'unhealthy',
    'cake': 'unhealthy',
    'sandwich': 'unhealthy',  # ⚠️ DUPLICATE KEY! (conflicts with 'healthy' above)
    'laptop': 'unhealthy',    # ⚠️ Non-food item for testing
    'cup': 'unhealthy'        # ⚠️ Non-food item for testing
    # ❌ 'bottle' removed
}
```

**Key Differences:**
- ❌ Your `'bottle': 'unhealthy'` was **removed**
- ⚠️ `'sandwich'` appears **twice** (once as 'healthy', once as 'unhealthy') - **CONFLICT!**
- ➕ Added `'laptop'` and `'cup'` as unhealthy (for testing)

---

## 2. food_detector.py - detect() Function

### Your Version (ab3889c):
```python
# ❌ NO detect() function in your version
# (detect_food_objects() exists with filter_unhealthy_food_only parameter)
```

### Merge Version (72a0238):
```python
def detect(image: Union[str, np.ndarray], 
           confidence_threshold: float = 0.5) -> Tuple[Tuple[int, int, int, int], str, str]:
    """
    Simple detection function that returns the first detected food item.
    Returns default values if no food is detected.
    ...
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
    return (0, 0, 100, 100), "none", "unknown"
```

**Key Differences:**
- ➕ Merge version **adds** `detect()` function (at line 227)
- ⚠️ **DUPLICATE** `detect()` function also added inside `if __name__ == "__main__"` block (line 435) - **INCORRECT INDENTATION!**

---

## 3. food_detector.py - if __name__ == "__main__" Block

### Your Version (ab3889c):
```python
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--webcam" or sys.argv[1] == "-w":
            detect_food(use_webcam=True)
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
```

### Merge Version (72a0238):
```python
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
    def detect(img):  # ⚠️ INCORRECTLY INDENTED - inside if __name__ block!
        detections = detect_food_objects(img, confidence_threshold=0.5, filter_food_only=True)
        if not detections:
            return (0, 0, 0, 0), "unknown", "unknown"
        det = detections[0]
        food_box = det['boundary']
        food_label = det['label']
        classification = det['classification']
        if classification == 'healthy':
            food_type = 'good'
        elif classification == 'unhealthy':
            food_type = 'bad'
        else:
            food_type = 'unknown'
        return food_box, food_label, food_type
```

**Key Differences:**
- ➕ Added comments about integration function
- ⚠️ **DUPLICATE** `detect()` function incorrectly indented inside `if __name__ == "__main__"` block

---

## 4. prickletickle.py - Imports

### Your Version (ab3889c):
```python
import cv2
import mediapipe as mp
import math
import requests
import pygame

from food_detector import detect_all_objects, detect_food_objects
```

### Merge Version (72a0238):
```python
import cv2
import mediapipe as mp
import math
import requests
import food_detector  # ➕ Added
import pygame

from food_detector import detect_all_objects, detect_food_objects
```

**Key Differences:**
- ➕ Added `import food_detector` (line 5)

---

## 5. prickletickle.py - is_bad_food() Function

### Your Version (ab3889c):
```python
# ❌ NO is_bad_food() function in your version
```

### Merge Version (72a0238):
```python
def is_bad_food(label, food_type=None):
    """
    Determine if food is bad/unhealthy
    Args:
        label: food label from detector
        food_type: optional explicit classification ("good" or "bad")
    """
    if food_type:
        return food_type.lower() == "bad"
    # Fallback to label-based classification
    return label.lower() in BAD_FOODS  # ⚠️ BAD_FOODS is NOT DEFINED!
```

**Key Differences:**
- ➕ Added `is_bad_food()` function
- ⚠️ **ERROR**: References `BAD_FOODS` which is **not defined** anywhere

---

## 6. New File: prickletickle-hand-detection.py

### Merge Version (72a0238) Only:
- ➕ **New file** added (84 lines)
- Appears to be a simpler version with hardcoded food box
- Not present in your commit

---

## Summary of Issues in Merge Version (72a0238)

### Critical Issues:
1. ❌ **Duplicate key in dictionary**: `'sandwich'` appears twice in `FOOD_CLASSIFICATION` (as both 'healthy' and 'unhealthy')
2. ❌ **Undefined variable**: `BAD_FOODS` referenced in `is_bad_food()` but never defined
3. ❌ **Duplicate function**: `detect()` function defined twice (once correctly, once incorrectly indented)
4. ❌ **Incorrect indentation**: `detect()` function inside `if __name__ == "__main__"` block

### Missing from Merge:
- ❌ Your `'bottle': 'unhealthy'` classification was removed

### Additions in Merge:
- ➕ `detect()` function (but duplicated)
- ➕ `is_bad_food()` function (but has undefined variable)
- ➕ `'laptop'` and `'cup'` as unhealthy (for testing)
- ➕ New file: `prickletickle-hand-detection.py`
- ➕ `import food_detector` in prickletickle.py

---

## Recommendations

1. **Fix duplicate 'sandwich' key** - decide if it should be 'healthy' or 'unhealthy'
2. **Remove duplicate `detect()` function** - keep only the one at module level
3. **Fix `BAD_FOODS` reference** - either define it or use `UNHEALTHY_FOODS` from food_detector
4. **Restore `'bottle': 'unhealthy'`** if needed
5. **Fix indentation** of comments in `if __name__ == "__main__"` block

