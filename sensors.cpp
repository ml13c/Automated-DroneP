// microcontroller (to be altered)
#include <iostream>
#define SENSOR_PIN 2
#define DANGER_ZONE_meters 1 // Example threshold distance in m
bool obstacleDetected = false;
int lastDistance = -1;
const char* PI_IP = "pi ip";//shouldnt change after configured
const int PI_PORT = 5005;
/*
this is intended for the flight controller so not final code just a draft
the flight controller will have a lidar and a 360 camera to detect obstacles
if the lidar detects something within a danger zone it will trigger a 'reverse'
same with the camera if it detects something like netting(main use)

*/
int readLidarDistance(){//maybe double or cm idk
    int distance = 5; // whatever is read from sensor
    // lidar reading logic here
    return distance; // replace with distance reading logic
}

// SEPERATE FROM 360 CAMERA TO DETECT GESTURES
bool readCameraObstacles(){// this is mainly for obstacles like nets that the lidar cant see
    // return true or false if obstacle matches thing like netting
    /*
    if obstacle matches netting photos/characteristics/data{
        return true;
    }
        data used for training should be nets from different distances, weather, lighting, angles
    */
    return false; // placeholder
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

// main flight loop (placeholder)
int main(){
    while(true){
        //setup stuff?
        if(checkObstacles()){
            obstacle_handler();
        }
        loop();//main flight loop
    }
}