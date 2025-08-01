# 🚀 Object Detection App

A powerful and user-friendly Python-based desktop application for real-time object detection. This GUI-driven app leverages the **YOLOv3 deep learning model** to perform detections on both **static images** and **live webcam feeds**, making advanced computer vision accessible.

---

## ✨ Features

* **Intuitive GUI:** Built with Tkinter for a seamless user experience.
* **Static Image Detection:** Analyze objects within your chosen images.
* **Real-Time Webcam Detection:** Experience live object detection directly from your webcam.
* **YOLOv3 Powered:** Utilizes the robust YOLOv3 model via OpenCV's Deep Neural Network (DNN) module for accurate and fast detections.
* **Confidence Scores:** Displays detected objects along with their confidence scores for detailed analysis.

---

## 📂 Project Structure
```tree
Object_Detection_App/
├── GUI.py                 # Main application script for the GUI
├── yolov3.cfg             # YOLOv3 model configuration file
├── yolov3.weights         # Pre-trained YOLOv3 weights (download separately)
├── coco.names             # Class labels for the COCO dataset
├── camera.ico             # Application icon
├── image/                 # Directory for sample input/test images
├── init.py            # Marks the directory as a Python package
└── README.md              # Project documentation
```

---

## ⚙️ Setup and Installation

### Requirements

Ensure you have the following installed:

* **Python 3.10** or later
* **OpenCV**
* **NumPy**
* **Tkinter** (usually included with Python on Windows; for Linux/macOS, you might need to install it separately, e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu).

### Installation Steps

1.  **Clone the Repository (or download the project files):**
    ```bash
    git clone https://github.com/AbdulAhadSiddiqui-0786/Object-Detection #
    cd Object_Detection_App
    ```

2.  **Install Dependencies:**
    ```bash
    pip install opencv-python numpy
    ```

3.  **Download YOLOv3 Weights:**
    The `yolov3.weights` file is essential for the model to function, but due to its large size, it's not included in the repository. Please download it manually and place it in the `Object_Detection_App/` root directory.

    **🔗 Download Link:** [yolov3.weights](https://drive.google.com/file/d/1oGc7rKSsG6kkhUerYaFmDk9tDsL8XNHV/view?usp=sharing)

---

## 🚀 How to Use

1.  **Run the Application:**
    Navigate to the `Object_Detection_App/` directory in your terminal and execute:
    ```bash
    python GUI.py
    ```

2.  **Object Detection in Images:**
    Click the "**Open Image**" button to select an image from your local machine and view the detected objects.

3.  **Real-Time Webcam Detection:**
    Click "**Start Real-Time Detection**" to activate your webcam and see objects detected live.

---

## 💡 About

This object detection application was developed by **Abdul Ahad and team**. It's designed to provide a straightforward yet powerful solution for leveraging YOLOv3 in a desktop environment.