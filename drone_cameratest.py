# gestures:
# # Pseudocode for a Raspberry Pi (e.g., Python)
# import detect_handgestures
# import RPi.GPIO as GPIO
# import time
# from threading import Thread
# import sensors
# # bridge for ai and flight controller
# # 
# #
# # 
# # gesture INTERRUPTS override modes if user wants to change commands
# # second thread is in charge of user tracking with camera #2 AI camera(180/-180)
# # 
# #

# # This function runs when a gesture interrupt is detected
# def command_handler(channel):
#     print("Hand gesture detected! Executing command...")
#     # This is where you would call a function to land the drone or take a picture
#     land_drone()

# # This is where the drone's primary flight logic runs
# def flightmode_follow_user():
#     while True:
#         # Constantly analyze camera feed to follow the user
#         track_user_with_camera()#this needs to
#         # Perform other high-level tasks
#         time.sleep(0.01)
# # follow user closer (more demanding cause needs to check distance from user)
# def flightmode_follow_user_close():
# # Set up the GPIO pin for the gesture recognition interrupt
# GPIO.setmode(GPIO.BCM)
# GESTURE_PIN = 17 # This pin is an output from the vision processing module

# # Set the pin as an input
# GPIO.setup(GESTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# # Attach the interrupt to the gesture recognition pin
# # The bouncetime prevents multiple interrupts from a single gesture
# GPIO.add_event_detect(GESTURE_PIN, GPIO.FALLING, callback=command_handler, bouncetime=200)

# # Run the main flight loop in a separate thread so it doesn't block
# flight_thread = Thread(target=flightmode_follow_user)
# flight_thread.start()

# # Keep the main program running to listen for interrupts
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     GPIO.cleanup()

##CODE ABOVE TO BE USED


#test code for functional camera 360 degree servo(170/-170) to not destroy cable
import cv2
import mediapipe as mp
import time
import socket

# #mediapipe hands model quick reference:
# 1,5,9,13,17 - knuckles
# thumb = 1-4
# index = 5-8
# middle = 9-12
# ring = 13-16
# pinky = 17-20

# #key points 
#  - camera will always be on looking for locked user and tracking them at all times
#  - specifically tracking if they are making any specific gestures to trigger commands
#  - user should be centered within frame/lens of camera(goal/circle) and drone moves to keep them centered

#mediapipe stuff
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)#min_tracking_confidence=0.5
mp_draw = mp.solutions.drawing_utils
# webcam/pi test code(SUBJECT TO CHANGE AFTER FIRST FLIGHT TEST)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
frame_center_x = 320
center_tolerance = 50
last_command = "hover"
last_gesture = None

print("Starting camera test")

while True:                                             #always looking for user/updating
    ret, frame = cap.read()
    command = "hover"  # default
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    
    if results.multi_hand_landmarks:  # if hand detected
        hand_landmarks = results.multi_hand_landmarks[0]#wrist is 0
        landmarks = hand_landmarks.landmark
        thumb = landmarks[2:5]
        index = landmarks[6:9]
        middle = landmarks[10:13]
        ring = landmarks[14:17]
        pinky = landmarks[18:21]
        
        def is_finger_extended(fingertip):
            return landmarks[fingertip].y < landmarks[fingertip - 2].y
        
        #hand check(might be wrong or need improvement tbh but it works!!)
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        index_mcp = landmarks[5]
        right_hand = thumb_cmc.x < index_mcp.x
        
        #IMPORTANT PART FOR GESTURE RECOGNITION - when doing a gesture thumb should always be as if you are making
        # a fist or else thumb is detected as extended which could be detected as 5 which does nothing(for now cause testing)
        
        #thumb handling(flipped if facing back of user but should be fine for now. detect back of user due to lack of facial features?)
        def is_thumb_extended():#hand use MIGHT BE REMOVED -------TBD-----
            if right_hand:
                return landmarks[4].x > landmarks[3].x
            else:
                return landmarks[4].x < landmarks[3].x#thumb goes right
            
        thumb_extended = is_thumb_extended()
        index_extended = is_finger_extended(8)
        middle_extended = is_finger_extended(12)
        ring_extended = is_finger_extended(16)
        pinky_extended = is_finger_extended(20)
        
        
        # GESTURES
        #thumb only - hover
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = 5
            #command = 5
            print("Gesture: 5 - hover")
            last_gesture = gesture#might need to change the order of when this is set(right before command is set?)
        #4 fingers no thumb
        if not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            gesture = 4
            #command = 4
            print("Gesture: 4 - return home")
            last_gesture = gesture
        #3 finger (index, middle, ring)
        if not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
            gesture = 3
            #command = 3
            print("Gesture: 3 - land")
            last_gesture = gesture
        #2 finger (index, middle)
        if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            gesture = 2
            #command = 2
            print("Gesture: 2 - take picture")
            last_gesture = gesture
        #1 finger (index)
        if not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = 1
            #command = 1
            print("Gesture: 1 - control mode(should be locked in this mode until otherwise specified)")
            last_gesture = gesture
            
            height, width, _ = frame.shape
            cx = int(hand_landmarks.lanmark[0].x * width)#center of palm
            if cx < frame_center_x - center_tolerance:
                command = "left"
            elif cx > frame_center_x + center_tolerance:
                command = "right"
            else:
                print("command set as hover(not 5)")
                command = "hover"#5
        else:
            gesture = None
            print("No valid gesture detected")
        if command != last_command:
            #sock.sendto(command.encode(), (UDP_IP, UDP_PORT))
            print(f"Command: {command}")
            last_command = command
            
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    cv2.imshow("Drone Camera POV Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()