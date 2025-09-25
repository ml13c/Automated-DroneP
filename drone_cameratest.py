gestures:
# Pseudocode for a Raspberry Pi (e.g., Python)
import detect_movement
import RPi.GPIO as GPIO
import time
from threading import Thread
import sensors
# bridge for ai and flight controller
# 
#
#
# gesture INTERRUPTS override modes if user wants to change commands
#
#
#

# This function runs when a gesture interrupt is detected
def command_handler(channel):
    print("Hand gesture detected! Executing command...")
    # This is where you would call a function to land the drone or take a picture
    land_drone()

# This is where the drone's primary flight logic runs
def flightmode_follow_user():
    while True:
        # Constantly analyze camera feed to follow the user
        track_user_with_camera()#this needs to
        # Perform other high-level tasks
        time.sleep(0.01)
# follow user closer (more demanding cause needs to check distance from user)
def flightmode_follow_user_close():
# Set up the GPIO pin for the gesture recognition interrupt
GPIO.setmode(GPIO.BCM)
GESTURE_PIN = 17 # This pin is an output from the vision processing module

# Set the pin as an input
GPIO.setup(GESTURE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Attach the interrupt to the gesture recognition pin
# The bouncetime prevents multiple interrupts from a single gesture
GPIO.add_event_detect(GESTURE_PIN, GPIO.FALLING, callback=command_handler, bouncetime=200)

# Run the main flight loop in a separate thread so it doesn't block
flight_thread = Thread(target=flightmode_follow_user)
flight_thread.start()

# Keep the main program running to listen for interrupts
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()

