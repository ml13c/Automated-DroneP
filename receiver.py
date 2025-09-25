import socket
import select
import threading
import time
# runs on pi
GESTURE_PORT = 5005
OBSTACLE_PORT = 5006
TRACKING_PORT = 5007

# flighr controller parametes
MAV_SERIAL = '/dev/ttyAMA0'
MAV_BAUD = 57600

# state
flight_mode = 'IDLE'  # Other modes: 'FOLLOW_USER', 'LAND', etc.
obstacle_state = False

def send_to_flight_controller(command):
    