import cv2
import mediapipe as mp
import socket

PI_IP = "192.168.50.8"  # Replace with your Pi's IP
PI_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height
frame_center_x = 320
center_tolerance = 50  # pixels for dead zone

last_command = "stop"

print("Starting hand tracking...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    command = "stop"

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

    # Only send if command changed
    if command != last_command:
        sock.sendto(command.encode(), (PI_IP, PI_PORT))
        print(f"Sending command: {command}")
        last_command = command

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
