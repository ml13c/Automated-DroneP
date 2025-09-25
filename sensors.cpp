// microcontroller (to be altered)
#include <iostream>
#define SENSOR_PIN 2
#define DANGER_ZONE_meters 1 // Example threshold distance in m
bool obstacleDetected = false;
int lastDistance = -1;

int readLidarDistance(){
    int distance = 5; // whatever is read from sensor
    // Placeholder function to simulate reading distance from a LIDAR sensor
    // lidar hardware interaction code would go here
    return distance; // replace-distance reading logic
}

// SEPERATE FROM 360 CAMERA TO DETECT GESTURES
void readCameraObstacles(){// this is mainly for obstacles like nets that the lidar cant see
    // return true or false if obstacle matches thing like netting
}
volatile bool checkObstacles(){
    int distance = readLidarDistance();
    lastDistance = distance;
    if (distance > 0 && distance < DANGER_ZONE_meters) { // temp threshold distance
        obstacleDetected = true;
        return true;
    }
    obstacleDetected = false;
    return false;
}
//Interrupt Service Routine (ISR) for lidar and camera combo
void obstacle_handler() {//maybe add parameter in future cause its better for testing but no object = faster i think
    std::cout << "Obstacle detected!" << std::endl;
    //if using multiple sensors check which one triggered the interrupt and reverse nearesest motor to avoid obstacle
    //likely going to only use one sensor and camera due to cost giving like fpv
    if(obstacleDetected){
        obstacle_reverse_drone();
    }
}

void obstacle_reverse_drone(){
    // Logic to reverse or avoid the obstacle
    std::cout << "Reversing to avoid obstacle." << std::endl;
    // This is a placeholder function. In a real implementation, this would control the drone's motors.
}
// void setup() {
//     // Set the sensor pin as an input
//     pinMode(SENSOR_PIN, INPUT);

//     // Attach the interrupt to the sensor pin
//     // CHANGE means the interrupt triggers on both rising and falling edges of the signal
//     attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), obstacle_handler, CHANGE);

//     // Start the motors and begin the flight loop
//     start_motors();
// }

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
int main(){
    while(true){
        //setup stuff?
        
        if(obstacleDetected()){
            obstacle_handler(obstacleDetected());
        }
    }
}