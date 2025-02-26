UNDERSTANDING HOW THE GESTURES ARE DETECTED
mediapipe_hand.png
The image I am refering to above gives a solid understanding of how one is able to calculate the gestures one signs.
With 21 nodes nodes starting at the wrist as 0, each node has its own index with a preset description of where it usually is.
***NOTE: In orer for live video mode to work efficiently the handn landmarker only retriggers if the model no longer identifies the presence of hands
within the frame.***