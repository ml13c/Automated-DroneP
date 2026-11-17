import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import socket


#socket stuff here
## this will likely change is it would be too slow honestly

# send closer or further(maybe might just use sensors for readings and use tracking of user for centering or both)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

lock_acquired = False
baseline_frames_needed = 8
baseline_areas = []
BASELINE_FRAMES = baseline_frames_needed

# smoothing params
alpha = 0.2
smoothed_area = None
threshold = 0.12

# keep last state to avoid spamming
last_state = "unknown"              # "closer", "further", "stable", "unknown"
last_state_time = 0

# torso confidence check
def torso_confident(lm):
    # use shoulders + hips visibility when available; if not available assume confident
    keys = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    vis = []
    for k in keys:
        v = lm[k].visibility if hasattr(lm[k], "visibility") else 1.0
        vis.append(v)
    return np.mean(vis)  # return average visibility (0..1)

print("Starting person tracking. Raise full hand (gesture 5) to lock on.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    hands_results = hands.process(frame_rgb)

    # first person who shows 5 fingers is locked in as user to track(temporary maybe ill use someting easier later..tracker?)
    if not lock_acquired and hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            def is_finger_up(tip_id):
                return lm[tip_id].y < lm[tip_id - 2].y

            # thumb check: approximate (works for many angles) by checking tip vs ip on x-axis relative to index mcp
            thumb_extended = (lm[4].x - lm[3].x) * (1 if lm[5].x > lm[2].x else -1) > 0  # rough
            idx = is_finger_up(8)
            mid = is_finger_up(12)
            rng = is_finger_up(16)
            pnk = is_finger_up(20)

            if all([thumb_extended, idx, mid, rng, pnk]):
                print("User detected! Locking to this person.")
                lock_acquired = True
                baseline_areas = []
                smoothed_area = None
                last_state = "unknown"
                break

    # Step 2: if locked, compute bounding box area and torso width
    if lock_acquired and pose_results.pose_landmarks:
        plm = pose_results.pose_landmarks.landmark

        # bounding box from all pose landmarks(px coords)
        xs = [int(l.x * w) for l in plm]
        ys = [int(l.y * h) for l in plm]
        x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
        y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))

        box_w = x_max - x_min
        box_h = y_max - y_min
        box_area = box_w * box_h + 1e-6  # avoid zero

        left_sh = plm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = plm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hp = plm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hp = plm[mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder_width = abs(left_sh.x - right_sh.x)
        hip_width = abs(left_hp.x - right_hp.x)
        avg_torso_width = (shoulder_width + hip_width) / 2.0 + 1e-6
        conf = torso_confident(plm)

        # Collect initial baseline frames to form stable reference
        if len(baseline_areas) < BASELINE_FRAMES:
            baseline_areas.append(box_area)
            # update a temporary smoothed_area during baseline collection
            if smoothed_area is None:
                smoothed_area = box_area
            else:
                smoothed_area = alpha * box_area + (1 - alpha) * smoothed_area

            cv2.putText(frame, f"Calibrating baseline ({len(baseline_areas)}/{BASELINE_FRAMES})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
            # draw current box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        else:
            # baseline established
            baseline_area = np.mean(baseline_areas)
            # update EMA only when torso is confident (less occlusion)
            if conf > 0.4:
                if smoothed_area is None:
                    smoothed_area = box_area
                else:
                    smoothed_area = alpha * box_area + (1 - alpha) * smoothed_area
            else:
                # if torso occluded, skip updating smoothed_area (prevents false updates)
                cv2.putText(frame, "Occluded - skipping update", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 2)

            # compare smoothed_area vs baseline_area
            if smoothed_area is not None:
                change_ratio = (smoothed_area - baseline_area) / baseline_area

                # only trigger when change beyond threshold
                if change_ratio <= -threshold:
                    state = "further"
                elif change_ratio >= threshold:
                    state = "closer"
                else:
                    state = "stable"

                # require state change to be different from last_state before printing
                if state != last_state:
                    now = time.time()
                    # small debounce: ignore very fast flips (<0.5s)
                    if now - last_state_time > 0.5:
                        if state == "further":
                            print("Person getting further")
                        elif state == "closer":
                            print("Person getting closer")
                        last_state = state
                        last_state_time = now

            # draw locked user box and info
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Locked user", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw torso width line (visual)
        sx1 = int(left_sh.x * w); sx2 = int(right_sh.x * w)
        cv2.line(frame, (sx1, int(left_sh.y * h)), (sx2, int(right_sh.y * h)), (255, 0, 0), 2)

        # safety: if person disappears for a while, unlock
        # if many landmarks are out of frame (y coord out of [0,1]) or low confidence, unlock
        out_of_frame_ratio = np.mean([1.0 if (l.x < 0 or l.x > 1 or l.y < 0 or l.y > 1) else 0.0 for l in plm])
        if conf < 0.2 or out_of_frame_ratio > 0.3:
            # release lock
            print("User lost/unreliable landmarks — unlocking.")
            lock_acquired = False
            baseline_areas = []
            smoothed_area = None
            last_state = "unknown"

    # show frame
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Robust Person Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
