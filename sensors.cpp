// microcontroller (to be altered)

//  Define the pin connected to the ultrasonic sensor's interrupt output
#define SENSOR_PIN 2

// Declare a global variable to store the obstacle status
volatile bool obstacleDetected = false;

//Interrupt Service Routine (ISR) for the ultrasonic sensor
void obstacle_handler() {
    // This code runs immediately when the interrupt is triggered
    obstacleDetected = true;
    // For a real drone, this would trigger a function to stop or turn
}

void setup() {
    // Set the sensor pin as an input
    pinMode(SENSOR_PIN, INPUT);

    // Attach the interrupt to the sensor pin
    // CHANGE means the interrupt triggers on both rising and falling edges of the signal
    attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), obstacle_handler, CHANGE);

    // Start the motors and begin the flight loop
    start_motors();
}

void loop() {
    // Main flight loop
    // This is the primary, continuous task of the microcontroller

    if (obstacleDetected) {
        // Stop the motors or execute an avoidance maneuver
        // For a real drone, this would call a more complex function
        stop_motors();
        // Reset the flag so it only runs once per detection
        obstacleDetected = false;
    }

    // Continue with regular flight calculations
    fly_straight();
}