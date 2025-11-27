import cv2
import mediapipe as mp
import math
import requests
import pygame

# Initialize pygame mixer for audio
pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Example: output from your colleague's detector
# Simulating: bounding box of fries (unhealthy)
food_box = (300, 200, 450, 350)   # xmin, ymin, xmax, ymax
food_label = "fries"

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
    
    # Draw food bounding box
    cv2.rectangle(img, (food_box[0], food_box[1]),
                       (food_box[2], food_box[3]), (0,255,0), 2)
    cv2.putText(img, food_label, (food_box[0], food_box[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

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

            # Check if finger is inside food box
            if point_in_box(px, py, food_box):

                # Detect a pinch/grab gesture
                if is_pinch(handLms) and food_label == "fries":
                    trigger_warning()

    cv2.imshow("Food Grab Detector", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
