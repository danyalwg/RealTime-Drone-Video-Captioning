# RealTime-Drone-Video-Captioning

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Detailed Architecture](#detailed-architecture)
    - [Module Descriptions](#module-descriptions)
    - [Code Walkthrough](#code-walkthrough)
4. [Installation and Setup](#installation-and-setup)
    - [Cloning the Repository](#cloning-the-repository)
    - [Setup on Ubuntu](#setup-on-ubuntu)
    - [Setup on Windows](#setup-on-windows)
5. [Usage Instructions](#usage-instructions)
6. [Fine-Tuning the BLIP Model](#fine-tuning-the-blip-model)
    - [Training Setup](#training-setup)
    - [Data Preparation and Augmentation](#data-preparation-and-augmentation)
    - [Model Adaptation and Training Process](#model-adaptation-and-training-process)
    - [Plot Generation, Annotation, and Analysis](#plot-generation-annotation-and-analysis)
7. [Advanced Features and Configurations](#advanced-features-and-configurations)
8. [Troubleshooting and FAQ](#troubleshooting-and-faq)

---

## Introduction

Welcome to the **RealTime-Drone-Video-Captioning** project! This repository houses a sophisticated Python application designed for real-time captioning of drone video feeds. Leveraging a fine-tuned BLIP model from Salesforce, the application provides accurate, context-aware captions for various video inputs including files, RTSP streams, and USB cameras. This guide is intended for developers, system administrators, and end-users who want to understand every detail of the project.

---

## Project Overview

The primary goal of this project is to deliver a robust, real-time video captioning solution with an intuitive GUI built using PyQt5. The system is divided into three core components:

- **Video Input Module:** Supports video files, RTSP streams, and USB cameras.
- **Captioning Module:** Utilizes a fine-tuned BLIP model to generate descriptive captions on the fly.
- **Log Monitoring Module:** A Flask-based web server displays real-time terminal logs.

---

## Detailed Architecture

### Module Descriptions

1. **User Interface (UI):**
   - **Framework:** Developed with PyQt5.
   - **Features:** Tab-based navigation for switching between video display and terminal logs.
   - **Components:** Video display area, caption label, control buttons (Start/Stop), input selectors, and performance metrics.

2. **Video Processing & Captioning:**
   - **Frame Capture:** Uses OpenCV to read frames from various video sources.
   - **Captioning:** A dedicated `QThread` (`CaptionThread`) processes frames and generates captions using the BLIP model.
   - **Metrics:** Real-time computation of video FPS and inference FPS.

3. **Log Server:**
   - **Technology:** Flask web server.
   - **Functionality:** Displays timestamped terminal logs with auto-refresh (every 5 seconds).
   - **Thread Safety:** Uses Python threading locks for concurrent log updates.

### Code Walkthrough

- **Main Application File (`app.py`):**
  - Initializes the PyQt5 application, applies a dark theme with qdarkstyle, and sets up the main window.
  - Loads the BLIP model and processor, selecting CUDA if available.
  - Manages separate threads for video capture, caption generation, and the Flask log server.

- **CaptionThread Class:**
  - Continuously retrieves the latest video frame and generates a caption.
  - Utilizes a sleep interval to balance performance and responsiveness.
  - Emits signals (`caption_generated`) to update the UI asynchronously.

- **Flask App:**
  - Hosts a web page at the root route ("/") that displays live terminal logs.
  - The HTML template includes a meta-refresh tag for auto-updating every 5 seconds.

---

## Installation and Setup

### Cloning the Repository

Open your terminal (or Command Prompt on Windows) and run:

```
git clone https://github.com/your-username/RealTime-Drone-Video-Captioning.git
cd RealTime-Drone-Video-Captioning
```

### Setup on Ubuntu

1. **Update System Packages:**

   ```
   sudo apt-get update
   sudo apt-get upgrade
   ```

2. **Install Python 3 and pip3:**

   ```
   sudo apt-get install python3 python3-pip
   ```

3. **Install NVIDIA Drivers and CUDA:**

   - Ensure your system has the appropriate NVIDIA drivers installed.
   - Download and install the CUDA toolkit from: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - For PyTorch GPU support, visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to get the correct installation command.

4. **Install Additional Dependencies:**

   ```
   sudo apt-get install libopencv-dev
   ```

5. **Install Python Dependencies:**

   ```
   pip3 install -r requirements.txt
   ```

### Setup on Windows

1. **Install Python 3:**
   - Download the latest Python 3 installer from [Python Downloads for Windows](https://www.python.org/downloads/windows/).
   - Run the installer and ensure that Python is added to your system PATH.

2. **Install NVIDIA Drivers and CUDA:**
   - Download and install the latest NVIDIA drivers and the CUDA toolkit from: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - For PyTorch GPU support, visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and follow the instructions for your configuration.
   - **Note:** C++ Build Tools are not required for this project.

3. **Install Python Dependencies:**

   Open Command Prompt and run:

   ```
   pip3 install -r requirements.txt
   ```

4. **Verify Installation:**

   ```
   python3 -c "import cv2, torch, PyQt5, flask, transformers"
   ```

---

## Usage Instructions

1. **Starting the Application:**

   Navigate to the repository folder and run:

   ```
   python3 app.py
   ```

2. **Interacting with the UI:**
   - **Main Tab:**  
     - Select a video source (File, RTSP, or USB Camera).
     - Click "Start" to begin video capture.
     - View real-time captions and performance metrics (video FPS and inference FPS).
   - **Terminal Tab:**  
     - Monitor real-time logs that capture every event (e.g., video source selection, caption generation).

3. **Accessing the Flask Log Server:**

   Open your web browser and navigate to `http://localhost:5000` to view live terminal logs.

---

## Fine-Tuning the BLIP Model

The BLIP model (Salesforce/blip-image-captioning-large) was fine-tuned to meet domain-specific requirements. The process enhanced the model’s caption quality and contextual accuracy.

### Training Setup

- **Base Model:** Salesforce/blip-image-captioning-large
- **Dataset:** A custom, domain-specific image-caption dataset curated from drone footage.
- **Hardware:** Utilized CUDA-enabled GPUs for accelerated training.
- **Hyperparameters:**
  - **Learning Rate:** 1e-5 (with a warm-up scheduler)
  - **Batch Size:** 16
  - **Epochs:** 500
  - **Optimizer:** AdamW with weight decay
  - **Loss Function:** Cross-Entropy Loss with regularization

### Data Preparation and Augmentation

- **Preprocessing:**  
  - Resize images to a standard resolution (384x384).
  - Normalize images using ImageNet statistics.
- **Data Augmentation:**  
  - Apply techniques such as random cropping, rotation, and color jitter.
  - These augmentations improve model robustness by simulating varied real-world conditions.

### Model Adaptation and Training Process

1. **Model Initialization:**
   - Load the pre-trained BLIP model and reconfigure its final layers to match the custom dataset’s vocabulary.
   
2. **Training Procedure:**
   - Train the model over 500 epochs.
   - Log key metrics at each epoch:
     - Training Loss
     - Validation Loss
     - BLEU, CIDEr, METEOR, and ROUGE scores

3. **Validation:**
   - Use a validation set to evaluate performance after each epoch.
   - The best epoch is marked based on a combination of loss and metric improvements.

### Plot Generation, Annotation, and Analysis

The training process generates several plots to monitor model performance. All plots are saved in the **plots** folder in the repository.

1. **BLEU Score vs Epoch**  
   - **Description:** Evaluates the quality of machine-generated translations.
   - **Trend:** Increases steadily, peaking around epoch 492 (score: 0.82).
   - **Key Takeaway:** Translation quality improves with training, stabilizing after ~epoch 400.

2. **CIDEr Score vs Epoch**  
   - **Description:** Measures the similarity between generated and reference captions.
   - **Trend:** Increases steadily, peaking at epoch 458 (score: 8.20).
   - **Key Takeaway:** Best captioning performance is achieved around epoch 450.

3. **METEOR Score vs Epoch**  
   - **Description:** Considers synonyms, stemming, and paraphrasing in translations.
   - **Trend:** Grows steadily, peaking at epoch 472 (score: 0.77).
   - **Key Takeaway:** Indicates continuous quality improvements until about epoch 470.

4. **ROUGE Score vs Epoch**  
   - **Description:** Measures recall overlap between model outputs and references.
   - **Trend:** Increases and peaks at epoch 396 (score: 0.83).
   - **Key Takeaway:** Summarization performance improves and stabilizes around epoch 400.

5. **Training Loss vs Epoch**  
   - **Description:** Shows how well the model fits the training data.
   - **Trend:** Decreases from ~3.0 to a minimum of 0.41 at epoch 472.
   - **Key Takeaway:** Lower training loss indicates better model fitting, with diminishing returns after epoch 470.

6. **Validation Loss vs Epoch**  
   - **Description:** Indicates how well the model generalizes to unseen data.
   - **Trend:** Decreases steadily, reaching 0.61 at epoch 478 before stabilizing.
   - **Key Takeaway:** Demonstrates good generalization with optimal performance around epoch 470–480.

---

## Advanced Features and Configurations

- **Multi-Source Video Input Handling:**  
  Supports video files, RTSP streams, and USB cameras with dynamic source detection.
  
- **Real-Time Metrics and Logging:**  
  Displays live video and inference FPS; logs are updated concurrently in both the UI and Flask web server.
  
- **Thread Safety and Concurrency:**  
  Uses dedicated threads (e.g., `CaptionThread`) for video capture and processing to ensure smooth UI performance.
  
- **Flask Server:**  
  Runs on a separate daemon thread to display terminal logs with automatic refresh.

---

## Troubleshooting and FAQ

### Common Issues

1. **Video Source Not Starting:**
   - Verify that the selected video file or stream URL is valid.
   - Ensure all required codecs and drivers are installed.

2. **Model Loading Errors:**
   - Check that the Transformers library is installed and updated.
   - Confirm GPU availability when using CUDA.

3. **Flask Server Inaccessible:**
   - Ensure no other service is using port 5000.
   - Review your system’s firewall settings.

### FAQ

- **Q:** What if the video freezes during processing?  
  **A:** This may be due to insufficient system resources. Try reducing the video resolution or using a more powerful machine.

- **Q:** How can I add support for additional video formats?  
  **A:** Update the file dialog filter in the `open_file` method within `app.py` to include the new formats.

- **Q:** Can I customize the fine-tuning parameters of the BLIP model?  
  **A:** Yes, adjust the hyperparameters in your training script as needed.

---

*End of README*
