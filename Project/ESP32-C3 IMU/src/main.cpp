#include <Arduino.h>
#include <Wire.h>
#include <esp_timer.h>
#include "MPU6050_6Axis_MotionApps20.h"

MPU6050 mpu;

uint8_t fifoBuffer[64];
uint16_t packetSize;
const uint8_t MPU_INT_PIN = 4;  // GPIO4

volatile bool mpuInterrupt = false;

Quaternion q;
VectorInt16 aa, gg;

// header bytes for alignment
const uint8_t H1 = 0xAA;
const uint8_t H2 = 0x55;

void IRAM_ATTR onMpuInterrupt() {
    mpuInterrupt = true;
}

void setup() {
    Serial.begin(115200);
    Wire.begin();

    mpu.initialize();
    mpu.dmpInitialize();

    // Calibration
mpu.setXAccelOffset(7274);
mpu.setYAccelOffset(3856);
mpu.setZAccelOffset(9610);
mpu.setXGyroOffset(-61);
mpu.setYGyroOffset(-82);
mpu.setZGyroOffset(7);

    mpu.setDMPEnabled(true);
    mpu.setIntEnabled(0x12);

    packetSize = mpu.dmpGetFIFOPacketSize();

    pinMode(MPU_INT_PIN, INPUT);
    attachInterrupt(digitalPinToInterrupt(MPU_INT_PIN), onMpuInterrupt, RISING);
}

void loop() {
    if (!mpuInterrupt) {
        return;
    }

    mpuInterrupt = false;

    if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer) == 0) {
        return;
    }

    const uint64_t sampleTimestampUs = static_cast<uint64_t>(esp_timer_get_time());

    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetAccel(&aa, fifoBuffer);
    mpu.dmpGetGyro(&gg, fifoBuffer);

    // send header
    Serial.write(H1);
    Serial.write(H2);

    // send host timestamp in microseconds
    Serial.write(reinterpret_cast<const uint8_t*>(&sampleTimestampUs), sizeof(sampleTimestampUs));

    // send quaternion (float)
    Serial.write(reinterpret_cast<const uint8_t*>(&q), sizeof(q));

    // send raw accel
    Serial.write(reinterpret_cast<const uint8_t*>(&aa), sizeof(aa));

    // send raw gyro
    Serial.write(reinterpret_cast<const uint8_t*>(&gg), sizeof(gg));
}