/*
for debugging refer to src code in : https://github.com/garmin/LIDARLite_RaspberryPi_Library/blob/master/src/lidarlite_v3.cpp
this is example/altered code from this repo and used to test lidar
with pi before full implementation.
*/
#include <cstdio>
#include <cstdint>
#include <csignal>
#include <chrono>
#include <thread>

// LIDAR-Lite header from the library you installed pi
//#include "lidarlite_v3.h"

static volatile bool keepRunning = true;

void handle_sigint(int)
{
    keepRunning = false;
}

//  LIDARLite_v3 myLidarLite;

int main()
{
    uint16_t distance = 0;
    uint8_t busyFlag = 1;
    std::signal(SIGINT, handle_sigint);

    // i2c peripheral initialize
    // NOTE: depending on the library version, i2c_init may return void or a status.
    myLidarLite.i2c_init();

    // Optionally configure LIDAR-Lite (0 == default settings)
   // myLidarLite.configure(0);

    while (keepRunning)
    {
        myLidarLite.takeRange();

        const int timeoutMs = 200;
        int waited = 0;
        const int pollIntervalMs = 10;
        while (true)
        {
            busyFlag = myLidarLite.getBusyFlag();
            if (busyFlag == 0)
                break;

            if (waited >= timeoutMs)
            {
                std::printf("Timeout waiting for LIDAR measurement\n");
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
            waited += pollIntervalMs;
        }
        if (busyFlag == 0)
        {
            distance = myLidarLite.readDistance();
            std::printf("Distance: %u cm\n", static_cast<unsigned int>(distance));
        }

        // Small delay to make output readable and avoid hammering the bus
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::printf("Exiting LIDAR test\n");
    return 0;
}