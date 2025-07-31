#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

BLEService customService("1234"); 
BLEStringCharacteristic statusCharacteristic("ABCD",                                              BLERead | BLENotify, 20);
BLEStringCharacteristic debugCharacteristic("ABCD", 
                                           BLERead | BLENotify, 20);

const int numSamples = 150;
int samplesRead = 0;
int connectionCount = 0;
unsigned long lastBlink = 0;
unsigned long lastNotify = 0;
bool ledState = false;

bool deviceConnected = false;
unsigned long connectionCheckInterval = 5000; 
unsigned long lastConnectionCheck = 0;

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = {
  "bad",
  "good"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  blinkLED(3, 100);

  Serial.println("Movement Classification with BLE");
  Serial.println("-------------------------------");

  Serial.println("Initializing BLE...");
  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
    }
  }
  Serial.println("BLE initialized successfully");

  statusCharacteristic.writeValue("Ready");
  debugCharacteristic.writeValue("Starting");

  BLE.setLocalName("Posture Police");
  BLE.setDeviceName("Posture Police");  
  BLE.setAdvertisedService(customService);
  customService.addCharacteristic(statusCharacteristic);
  customService.addCharacteristic(debugCharacteristic);
  BLE.addService(customService);

  BLE.setAdvertisingInterval(100);
  
  BLE.advertise();
  Serial.println("BLE Peripheral is advertising as 'nano33ble'...");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1) {
      blinkLED(2, 500);
    }
  }
  Serial.println("IMU initialized successfully");

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  blinkLED(5, 200); 
}

void loop() {
  BLE.poll();
  
  BLEDevice central = BLE.central();

  if (central) {
    if (!deviceConnected) {
      connectionCount++;
      deviceConnected = true;
      Serial.print("Connected to central: ");
      Serial.println(central.address());
      digitalWrite(LED_BUILTIN, HIGH);
      
      String debugMsg = "Connected #" + String(connectionCount);
      debugCharacteristic.writeValue(debugMsg);
      
      statusCharacteristic.writeValue("Ready");
      Serial.println("Sent initial Ready status");
    }

    processMotionData(central);
    
    if (millis() - lastConnectionCheck > connectionCheckInterval) {
      if (!central.connected()) {
        Serial.println("Connection lost");
        deviceConnected = false;
        digitalWrite(LED_BUILTIN, LOW);
      }
      lastConnectionCheck = millis();
    }
  } 
  else {

    if (deviceConnected) {
      Serial.println("Disconnected from central");
      deviceConnected = false;
      digitalWrite(LED_BUILTIN, LOW);

      BLE.advertise();
    }
    
    processMotionData(BLEDevice());
    
    if (millis() - lastBlink > 2000) {
      ledState = !ledState;
      digitalWrite(LED_BUILTIN, ledState);
      lastBlink = millis();
      

      BLE.advertise();
      Serial.println("Re-advertising BLE service...");
    }
  }
}

void processMotionData(BLEDevice central) {
  float aX, aY, aZ, gX, gY, gZ;


  if (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {

      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      if (samplesRead < 3) {
        Serial.print("Sample ");
        Serial.print(samplesRead);
        Serial.print(": aX=");
        Serial.print(aX);
        Serial.print(", aY=");
        Serial.print(aY);
        Serial.print(", aZ=");
        Serial.println("...");
      }

      int index = samplesRead * 6;
      tflInputTensor->data.f[index+0] = aX;
      tflInputTensor->data.f[index+1] = aY;
      tflInputTensor->data.f[index+2] = aZ;
      tflInputTensor->data.f[index+3] = gX;
      tflInputTensor->data.f[index+4] = gY;
      tflInputTensor->data.f[index+5] = gZ;
      samplesRead++;
      
    
      if (samplesRead % 20 == 0) {
        Serial.print("Collecting: ");
        Serial.print(samplesRead);
        Serial.print("/");
        Serial.println(numSamples);
      }
    }
  }


  if (samplesRead == numSamples) {

    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Model inference failed!");
      samplesRead = 0; 
      return;
    }


    Serial.println("\n--- CLASSIFICATION ---");
    for (int i = 0; i < NUM_GESTURES; i++) {
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 6);
    }
    Serial.println("---------------------------\n");

    String statusMessage;
    float probability = tflOutputTensor->data.f[1];

    if (tflOutputTensor->data.f[1] >= 0.5) {
     statusMessage = "Good";
    } else {
     statusMessage = "Bad";
;
    }
    
    Serial.print("*** DETECTED ACTIVITY: ");
    Serial.print(statusMessage);
    Serial.println(" ***");
    
    if (central) {
      statusCharacteristic.writeValue(statusMessage.c_str());
      
      String debugMsg = statusMessage + " " + String(millis()/1000) + "s";
      debugCharacteristic.writeValue(debugMsg);
      Serial.print("BLE notification sent: ");
      Serial.println(debugMsg);
    }

    samplesRead = 0;
    Serial.println("Starting new sample collection...");
  }
}

void blinkLED(int times, int duration) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(duration);
    digitalWrite(LED_BUILTIN, LOW);
    delay(duration);
  }
}


