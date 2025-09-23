# import cv2
# import mediapipe as mp
# import time
# prev_x = None
# swipe_start_time = None
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# # Start webcam
# cap = cv2.VideoCapture(0)

# '''
# Gesture features: this keeps track of what gesture is happening for the current frame
# ***Note that just because a gesture is logged for one frame it will not be logged for the next frame**
# so in order for an action to perfrom after a gesture the gesture should be logged for a certain number of frames
# ***THIS WILL CHANGE DEPENDING ON CIRCUMSTANCES/ENVIRONMENT***
# '''
# '''
# For now gestures are calculated on a fixed camera position and assuming the person is standing upright since
# the gestures are calculates based on the position of nodes. THIS WILL CHANGE SOON
# '''
# def fingers_up(hand_landmarks):
#     fingers = []

#     # Thumb (special case: check x-coordinates)
#     fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

#     # Other fingers (compare tip y-coordinate with middle joint y-coordinate)
#     for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
#         fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y else 0)

#     return fingers
# '''
# swipe gestures
# '''
# def detect_swipe(hand_landmarks):
#     global prev_x, swipe_start_time

#     current_x = hand_landmarks.landmark[9].x  # Use palm center

#     if prev_x is None:
#         prev_x = current_x
#         return None

#     movement = current_x - prev_x  # Difference in x-position
#     prev_x = current_x  # Update previous position

#     if abs(movement) > 0.05:  # Tune threshold for sensitivity
#         direction = "Right" if movement > 0 else "Left"

#         # Ensure it's a continuous movement, not jitter
#         if swipe_start_time is None:
#             swipe_start_time = time.time()

#         if time.time() - swipe_start_time > 0.3:  # Tune time threshold
#             print(f"Swipe {direction} detected!")
#             swipe_start_time = None  # Reset    

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip image for natural mirroring
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame for hand detection
#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             # if  fingers_up(hand_landmarks) == [0, 1, 1, 1, 1]:
#             #     print("fingers up function 4")
#             for hand_landmarks in results.multi_hand_landmarks:
#                 detect_swipe(hand_landmarks)

#     cv2.imshow("Hand Tracking", frame)
    



#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import time

# --- Hand tracking setup ---
prev_x = None
swipe_start_time = None
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Pose tracking setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    fingers = []
    fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)
    for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y else 0)
    return fingers

def detect_swipe(hand_landmarks):
    global prev_x, swipe_start_time
    current_x = hand_landmarks.landmark[9].x
    if prev_x is None:
        prev_x = current_x
        return None
    movement = current_x - prev_x
    prev_x = current_x
    if abs(movement) > 0.05:
        direction = "Right" if movement > 0 else "Left"
        if swipe_start_time is None:
            swipe_start_time = time.time()
        if time.time() - swipe_start_time > 0.3:
            print(f"Swipe {direction} detected!")
            swipe_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Hand detection ---
    results_hands = hands.process(rgb_frame)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_swipe(hand_landmarks)

    # --- Body tracking (Pose) ---
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get left and right shoulder positions
        left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate person's horizontal center in the frame
        person_center_x = (left_shoulder.x + right_shoulder.x) / 2

        # Get frame center
        frame_center_x = 0.5  # Normalized coordinates (0 to 1)

        # Define how far they can move before "camera moves"
        tolerance = 0.15  # 15% of frame width

        if person_center_x < frame_center_x - tolerance:
            print("Camera begins to move LEFT here")
        elif person_center_x > frame_center_x + tolerance:
            print("Camera begins to move RIGHT here")

    cv2.imshow("Hand + Body Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
