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
last_gesture = None

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

        thumb = landmarks[2:5]
        index = landmarks[6:9]
        middle = landmarks[10:13]
        ring = landmarks[14:17]
        pinky = landmarks[18:21]

        # determine if finger is up or down by comparing tip values to joints
        def is_finger_up(tip_id):
            # tip higher (smaller y) than lower joint means finger is up
            return landmarks[tip_id].y < landmarks[tip_id - 2].y

        # detect which hand we’re dealing with (important for thumb logic)
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        index_mcp = landmarks[5]
        right_hand = thumb_cmc.x < index_mcp.x  # thumb is on the left → right hand

        # special handling for thumb (horizontal orientation)
        def is_thumb_up():
            if right_hand:
                return landmarks[4].x < landmarks[3].x  # thumb extended left
            else:
                return landmarks[4].x > landmarks[3].x  # thumb extended right

        thumb_up = is_thumb_up()
        index_up = is_finger_up(8)
        middle_up = is_finger_up(12)
        ring_up = is_finger_up(16)
        pinky_up = is_finger_up(20)

        # gesture logic
        if thumb_up and not (index_up or middle_up or ring_up or pinky_up):  # thumb only
            gesture = "thumb"
            if gesture != last_gesture:
                print("thumb only means land")
                last_gesture = gesture

        elif (index_up and middle_up and ring_up and pinky_up) and not thumb_up:  # 4 fingers no thumb
            gesture = "4_fingers"
            if gesture != last_gesture:
                print("4 fingers detected")
                print("")
                last_gesture = gesture

        elif (index_up and middle_up and ring_up) and not (thumb_up or pinky_up):  # 3 fingers
            gesture = "3_fingers"
            if gesture != last_gesture:
                print("3 fingers detected")
                print("")
                last_gesture = gesture

        elif (index_up and middle_up) and not (thumb_up or ring_up or pinky_up):  # 2 fingers
            gesture = "2_fingers"
            if gesture != last_gesture:
                print("2 fingers detected")
                last_gesture = gesture

        elif index_up and not (thumb_up or middle_up or ring_up or pinky_up):  # 1 finger
            gesture = "1_finger"
            if gesture != last_gesture:
                print("1 finger detected. controlling drone now")
                last_gesture = gesture

            # control using hand movement for navigating drone (1 finger)
            h, w, _ = frame.shape
            cx = int(hand_landmarks.landmark[0].x * w)  # palm center

            if cx < frame_center_x - center_tolerance:
                command = "left"
            elif cx > frame_center_x + center_tolerance:
                command = "right"
            else:
                command = "stop"

        else:
            gesture = "none"
            print("no gesture detected")

        # only send command if changed
        if command != last_command:
            sock.sendto(command.encode(), (PI_IP, PI_PORT))
            print(f"Sending command: {command}")
            last_command = command

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand gesture test 2", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
