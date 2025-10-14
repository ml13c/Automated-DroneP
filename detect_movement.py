import cv2
import mediapipe as mp
import socket
# this is to detect hand gestures through media pipe
# later will use optimized model on pi camera but for now testing w stock

PI_IP = "192.168.50.8"  # Replace with your Pi's IP
PI_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# webcam(desktop)/ pi ai camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height
frame_center_x = 320
center_tolerance = 50  # pixels for dead zone

last_command = "stop"


# camera should always be on looking for user (specifically user hand) and trying to detect gestures

print("Starting hand tracking...")
# for quick reference(inclusive):
# 1,5,9,13,17 = joint knuckles parts of fingers
# thumb = 1-4
# index = 5-8
# middle = 9-12
# ring = 13-16
# pinky = 17-20
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    command = "stop"  # default command
    
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
    
    
    
    
        
        if landmarks[1:4]:#thumb
            print("thumb only means land")
        if landmarks[5:20]:# 4 fingers no thumb
            print("4 fingers detected")
        if landmarks[5:16]:# 3 fingers no thumb, no pinky
            print("3 fingers detected")
        if landmarks[5:12]:# 2 middle and index fingers
            print("2 fingers detected")
        if landmarks[5:8]:# 1 index finger
            print("1 finger detected. controlling drone now")
        # control using hand movement for navigating drone(1 index finger triggers this)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                cx = int(hand_landmarks.landmark[0].x * w)  # palm center

                # Dead zone
                if cx < frame_center_x - center_tolerance:
                    command = "left"
                elif cx > frame_center_x + center_tolerance:
                    command = "right"
                else:
                    command = "stop"

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # IMPORTANT: only sends to pi if command changes. stops flooding
        # and will be used later on in project for handling other sensors.
        if command != last_command:
            sock.sendto(command.encode(), (PI_IP, PI_PORT))
            print(f"Sending command: {command}")
            last_command = command

        cv2.imshow("Hand gesture test 2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
