# Posture Police

## Description

This project focuses on improving health by addressing poor sitting posture, a common issue that can lead to spinal dysfunction, chronic neck pain, and back pain.  

### **Tech Stack**
- **Flutter** – Mobile application development  
- **TensorFlow** – Machine learning model for posture classification  
- **Bluetooth Low Energy (BLE)** – Wireless communication with sensors  
- **Arduino Nano** – Data collection from accelerometer and gyroscope sensors  

### **Key Features**
1. **Minimal and Intuitive UI** – Effortlessly connect to sensors, recalibrate posture settings, and access posture tips.  
2. **Real-Time Posture Monitoring** – Detects and alerts users about poor posture instantly.  
3. **Streak & Habit Tracking** – Encourages long-term healthy posture habits through progress tracking and streak rewards.  


## Machine Learning

### Data Collection and Preprocessing 

We collected data using an **Arduino Nano** microcontroller equipped with an **inertial measurement unit (IMU)** sensor which provides 
- **Accelerometer readings** (X, Y, Z axes)  
- **Gyroscope readings** (X, Y, Z axes)  

These sensors captured the user's body orientation and movement in real time.

A total of **2,000 data points** were collected:
- **1,000 samples** for each posture class
- **20 time steps** per data point to capture temporal patterns, improving classification robustness

The dataset was randomly split into:
- **60% Training** – for learning model parameters
- **20% Validation** – for tuning hyperparameters and preventing overfitting
- **20% Testing** – for evaluating final model performance on unseen data




### Model Specification

The classification model is built using **TensorFlow Keras** with an architecture designed for time-series sensor data.

#### Architecture
- **Input Layer:** Accepts segmented IMU sensor data (accelerometer and gyroscope)  
- **GlobalMaxPool1D Layer:** Aggregates temporal features by selecting maximum values across time steps  
- **Dense Layer (32 units, ReLU):** Learns nonlinear feature representations  
- **Output Layer (Softmax):** Produces class probabilities for each posture (`NUM_POSTURES`)

#### Training Configuration
- **Optimizer:** Adam (`learning_rate = 0.005`)  
- **Loss Function:** Sparse Categorical Crossentropy
- **Evaluation Metric:** Accuracy  
- **Batch Size:** 32  
- **Epochs:** 20  
- **Validation Data:** Separate validation dataset used for hyperparameter tuning and overfitting prevention  
- **Shuffling:** Enabled during training for better generalization

#### Model Summary

| Layer (type)            | Output Shape   | Param # |
|-------------------------|----------------|---------|
| GlobalMaxPooling1D      | (None, 6)      | 0       |
| Dense (32, ReLU)        | (None, 32)     | 224     |
| Dense (Softmax Output)  | (None, 2)      | 66      |

**Total parameters:** 290  
**Trainable parameters:** 290  
**Non-trainable parameters:** 0


### Model Conversion for Arduino Deployment
To deploy the trained posture classification model onto the Arduino Nano, we converted the TensorFlow model into a **TensorFlow Lite (TFLite)** format and then generated a C++ header file compatible with the Arduino environment.


## Acknowledgement


I would like to sincerely thank my teammates for their invaluable support and collaboration throughout this project. Their dedication, insights, and teamwork greatly contributed to the successful development and completion of Posture Police.
