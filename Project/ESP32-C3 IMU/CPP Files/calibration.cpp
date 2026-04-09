#include <Arduino.h>
#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050.h"

MPU6050 mpu;

void setup() {
    delay(2000);  // allow time for serial monitor to open
    Wire.begin();
    Wire.setClock(400000);
    Serial.begin(921600);
    while (!Serial);

    mpu.initialize();

    Serial.println("---------------------------------------");
    Serial.println("  MPU6050 AUTO-CALIBRATION SKETCH");
    Serial.println("---------------------------------------");
    // Serial.println(mpu.testConnection() ? "Connected to MPU6050" : "Connection failed");
    Serial.println("\nDO NOT MOVE THE SENSOR. Keep it perfectly flat on a table.");
    Serial.println("Send any character to start...");

    // Wait for user input
    while (Serial.available() && Serial.read()); // flush input buffer
    while (!Serial.available());                 // wait for input
    while (Serial.available() && Serial.read()); // flush again

    Serial.println("\nCalibrating... (This takes ~10-20 sec)");

    // Reset offsets first
    mpu.setXAccelOffset(0);
    mpu.setYAccelOffset(0);
    mpu.setZAccelOffset(0);
    mpu.setXGyroOffset(0);
    mpu.setYGyroOffset(0);
    mpu.setZGyroOffset(0);

    // Run built-in calibration
    mpu.CalibrateAccel(6);  // 6 loops
    mpu.CalibrateGyro(6);

    // Print results for use in main sketch
    Serial.println("\nCALIBRATION DONE!");
    Serial.println("---------------------------------------");
    Serial.print("mpu.setXAccelOffset("); Serial.print(mpu.getXAccelOffset()); Serial.println(");");
    Serial.print("mpu.setYAccelOffset("); Serial.print(mpu.getYAccelOffset()); Serial.println(");");
    Serial.print("mpu.setZAccelOffset("); Serial.print(mpu.getZAccelOffset()); Serial.println(");");
    Serial.print("mpu.setXGyroOffset(");  Serial.print(mpu.getXGyroOffset());  Serial.println(");");
    Serial.print("mpu.setYGyroOffset(");  Serial.print(mpu.getYGyroOffset());  Serial.println(");");
    Serial.print("mpu.setZGyroOffset(");  Serial.print(mpu.getZGyroOffset());  Serial.println(");");
    Serial.println("---------------------------------------");

    // Verify calibration: print a few readings
    Serial.println("\nVerifying accuracy (Ax/Ay ≈ 0, Az ≈ 16384, Gx/Gy/Gz ≈ 0)");
    for(int i = 0; i < 20; i++) {
        Serial.print("Ax: "); Serial.print(mpu.getAccelerationX());
        Serial.print("\tAy: "); Serial.print(mpu.getAccelerationY());
        Serial.print("\tAz: "); Serial.print(mpu.getAccelerationZ());
        Serial.print("\tGx: "); Serial.print(mpu.getRotationX());
        Serial.print("\tGy: "); Serial.print(mpu.getRotationY());
        Serial.print("\tGz: "); Serial.println(mpu.getRotationZ());
        delay(50);
    }
}

void loop() {
    // nothing to do here
}