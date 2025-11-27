import cv2
import mediapipe as mp
import math
import requests
import pygame

from food_detector import detect_all_objects, detect_food_objects

# Initialize pygame mixer for audio
pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize food items list and food boxes list
# food_items: list of dicts with 'boundary', 'label', 'classification', 'confidence'
# food_boxes: list of tuples (xmin, ymin, xmax, ymax) for quick access
# Initialize with one box for demonstration
initial_box = (300, 200, 450, 350)
food_items = [{'boundary': initial_box, 'label': 'fries', 'classification': 'unhealthy', 'confidence': 0.0}]
food_boxes = [initial_box]  # Initialize with one box (xmin, ymin, xmax, ymax)

def is_pinch(hand_landmarks):
    # Thumb tip: 4, Index tip: 8
    x1, y1 = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
    x2, y2 = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
    dist = math.hypot(x1 - x2, y1 - y2)
    return dist < 0.05    # threshold

def point_in_box(px, py, box):
    xmin, ymin, xmax, ymax = box
    return xmin <= px <= xmax and ymin <= py <= ymax

def trigger_warning():
    print("⚠️ Unhealthy food grab detected!")
    try:
        pygame.mixer.music.load("warning.mp3")
        pygame.mixer.music.play()
    except:
        print("Warning sound file not found or unable to play")

    # Trigger haptics via your web interface
    try:
        requests.post("http://localhost:8000/haptics", json={"type": "warning"}, timeout=1)
    except:
        print("Unable to reach haptics server")

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

while True:
    success, img = cap.read()
    if not success:
        break
    
    h, w, _ = img.shape
    
    # Run food detection once per frame (filter unhealthy food only)
    detections = detect_food_objects(img, filter_unhealthy_food_only=True)
    
    # Update food_items and food_boxes from detections
    if detections:
        # Update with new detections
        food_items = detections
        food_boxes = [item['boundary'] for item in food_items]
    else:
        # No detections found - keep last food_items and food_boxes values
        if not food_items:
            print("⚠️ No unhealthy food detected yet. Waiting for detection...")
        else:
            print("⚠️ No unhealthy food detected in current frame. Using last known food positions.")
    
    # Draw all food bounding boxes
    for item in food_items:
        x1, y1, x2, y2 = item['boundary']
        label = item['label']
        # Color: Red for unhealthy food
        color = (0, 0, 255)  # Red
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mediapipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip position in pixel coords
            finger_tip = handLms.landmark[8]
            px, py = int(finger_tip.x * w), int(finger_tip.y * h)

            cv2.circle(img, (px, py), 10, (255, 0, 0), -1)

            # Check if finger is inside any food box
            for i, box in enumerate(food_boxes):
                if point_in_box(px, py, box):
                    # Detect a pinch/grab gesture
                    if is_pinch(handLms):
                        # Since we're filtering unhealthy food only, trigger warning for all collisions
                        trigger_warning()
                        break  # Only trigger once per frame

    cv2.imshow("Food Grab Detector", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
