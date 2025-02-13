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
    - [Plot Generation and Annotation](#plot-generation-and-annotation)
7. [Advanced Features and Configurations](#advanced-features-and-configurations)
    - [Multi-Source Video Input Handling](#multi-source-video-input-handling)
    - [Real-Time Metrics and Logging](#real-time-metrics-and-logging)
    - [Thread Safety and Concurrency](#thread-safety-and-concurrency)
8. [Troubleshooting and FAQ](#troubleshooting-and-faq)
9. [Repository Structure and Contribution Guidelines](#repository-structure-and-contribution-guidelines)
10. [Performance Metrics and System Requirements](#performance-metrics-and-system-requirements)
11. [Future Enhancements](#future-enhancements)
12. [Appendices and References](#appendices-and-references)

---

## Introduction

Welcome to the **RealTime-Drone-Video-Captioning** project! This repository houses a sophisticated Python application designed for real-time captioning of drone video feeds. Leveraging a fine-tuned BLIP model from Salesforce, the application provides accurate, context-aware captions for various video inputs including files, RTSP streams, and USB cameras. This document is a comprehensive guide intended for developers, system administrators, and end-users interested in understanding every facet of the project.

---

## Project Overview

The primary goal of this project is to deliver a robust, real-time video captioning solution with an intuitive GUI built on PyQt5. The system comprises three major components:

- **Video Input Module:** Supports multiple sources (video file, RTSP, and USB camera).
- **Captioning Module:** Utilizes a fine-tuned BLIP model to generate descriptive captions on the fly.
- **Log Monitoring Module:** A Flask-based web server that displays real-time terminal logs.

The design emphasizes high performance, modularity, and thread-safe operations, ensuring seamless integration even under resource-constrained environments.

---

## Detailed Architecture

### Module Descriptions

1. **User Interface (UI):**
   - **Framework:** Built using PyQt5.
   - **Features:** Tab-based navigation for switching between the main video display and the terminal log.
   - **Components:** Video display area, caption label, control buttons (Start/Stop), input selectors, and metrics display.

2. **Video Processing & Captioning:**
   - **Frame Capture:** Utilizes OpenCV for reading frames from various video sources.
   - **Captioning:** A dedicated `QThread` (`CaptionThread`) is used to process frames and generate captions using the BLIP model.
   - **Metrics Calculation:** Real-time computation of video FPS and inference FPS.

3. **Log Server:**
   - **Technology:** Flask web server.
   - **Functionality:** Provides an auto-refreshing webpage displaying timestamped terminal logs.
   - **Thread Safety:** Uses Python threading locks to ensure safe concurrent access to log data.

### Code Walkthrough

Below is a high-level overview of the key code components:

- **Main Application File (`app.py`):**
  - **Initialization:** Sets up the PyQt5 application, applies a dark theme using qdarkstyle, and initializes the main window.
  - **Model Loading:** Loads the BLIP model and processor, determining the computing device (CUDA if available).
  - **Thread Management:** Manages separate threads for video processing and Flask server operation.

- **CaptionThread Class:**
  - **Purpose:** Continuously retrieves the latest frame and processes it to generate a caption.
  - **Mechanism:** Uses a loop with a sleep interval (`msleep`) to balance performance and responsiveness.
  - **Signal Emission:** Uses Qt signals (`caption_generated`) to update the UI asynchronously.

- **Flask App:**
  - **Route:** The root route ("/") renders an HTML template displaying real-time logs.
  - **Auto-Refresh:** The template includes a meta-refresh tag set to refresh every 5 seconds.

For an in-depth look at the code, refer to the inline comments and function docstrings within `app.py`.

---

## Installation and Setup

### Cloning the Repository

To clone the repository, open your terminal (or Command Prompt on Windows) and execute the following commands:

```
git clone https://github.com/your-username/RealTime-Drone-Video-Captioning.git
cd RealTime-Drone-Video-Captioning
```

### Setup on Ubuntu

1. **Update System Packages:**

   Open your terminal and run:
   
   ```
   sudo apt-get update
   sudo apt-get upgrade
   ```

2. **Install Python 3 and pip3:**

   ```
   sudo apt-get install python3 python3-pip
   ```

3. **Install Additional Dependencies:**
   
   Some system libraries may be required for OpenCV:
   
   ```
   sudo apt-get install libopencv-dev
   ```
   
   Refer to the [OpenCV Ubuntu Installation Guide](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) for more details.

4. **Install Python Dependencies:**

   ```
   pip3 install -r requirements.txt
   ```

### Setup on Windows

1. **Install Python 3:**
   
   - Download the latest Python 3 installer from [Python Downloads for Windows](https://www.python.org/downloads/windows/).
   - Run the installer and ensure you select the option to add Python to your PATH.

2. **Install Visual C++ Build Tools:**
   
   Some packages (e.g., PyQt5) may require additional build tools. If prompted, follow the on-screen instructions to install them.

3. **Install Python Dependencies:**

   Open Command Prompt and run:
   
   ```
   pip3 install -r requirements.txt
   ```

4. **Verify Installation:**
   
   To check if all packages are installed correctly, you can run:
   
   ```
   python3 -c "import cv2, torch, PyQt5, flask, transformers"
   ```

---

## Usage Instructions

1. **Starting the Application:**
   
   Navigate to the repository folder and run the application using:
   
   ```
   python3 app.py
   ```

2. **Interacting with the UI:**
   
   - **Main Tab:**  
     - Select a video source (File, RTSP, or USB Camera).
     - Click "Start" to begin video capture.
     - Real-time captions and metrics (video FPS and inference FPS) will be displayed.
   
   - **Terminal Tab:**  
     - View real-time logs capturing every event (e.g., video source selection, caption generation).

3. **Accessing the Flask Log Server:**
   
   Open a web browser and navigate to `http://localhost:5000` to view live terminal logs with an auto-refresh every 5 seconds.

---

## Fine-Tuning the BLIP Model

The BLIP model (Salesforce/blip-image-captioning-large) was fine-tuned to cater to domain-specific requirements, ensuring enhanced caption quality and contextual accuracy. Below is an exhaustive breakdown of the fine-tuning process.

### Training Setup

- **Base Model:** Salesforce/blip-image-captioning-large
- **Dataset:** A custom, domain-specific image-caption dataset curated from drone footage.
- **Hardware:** Utilized high-end GPUs (CUDA-enabled) to expedite training.
- **Hyperparameters:**
  - **Learning Rate:** 1e-5 (with warm-up scheduler)
  - **Batch Size:** 16
  - **Epochs:** 500
  - **Optimizer:** AdamW with weight decay
  - **Loss Function:** Cross-Entropy Loss with regularization

### Data Preparation and Augmentation

- **Preprocessing:**
  - All images were resized to a standard resolution (384x384) for consistency.
  - Normalization was performed using ImageNet statistics.
- **Data Augmentation:**
  - Techniques such as random cropping, rotation, and color jitter were applied.
  - Augmented data improved robustness by simulating various real-world conditions.

### Model Adaptation and Training Process

1. **Model Initialization:**
   - The pre-trained BLIP model was loaded and its final layers were reconfigured to accommodate the custom dataset's vocabulary.
   
2. **Training Procedure:**
   - The model was trained over 500 epochs.
   - At every epoch, the following metrics were logged:
     - Training Loss
     - Validation Loss
     - BLEU, CIDEr, METEOR, and ROUGE scores
     
3. **Validation:**
   - A separate validation set was used to evaluate the model’s performance after each epoch.
   - The best-performing epoch was marked based on a combination of loss and metric improvements.

### Plot Generation and Annotation

- **Visualization:**
  - Loss and metric curves were generated using Matplotlib.
  - Noise reduction techniques were applied to smooth the curves.
- **Annotations:**
  - Each plot includes arrows pointing to the best performance points.
  - Epoch and metric values are clearly labeled for in-depth analysis.

---

## Advanced Features and Configurations

### Multi-Source Video Input Handling

- **File Input:**  
  Users can select local video files in formats such as MP4, AVI, and MOV.
- **RTSP Streams:**  
  Allows live streaming from RTSP-compatible cameras.
- **USB Camera:**  
  The application dynamically detects available USB cameras and populates a dropdown list for selection.

### Real-Time Metrics and Logging

- **Performance Metrics:**
  - **Video FPS:** Calculated by counting frames over elapsed time.
  - **Inference FPS:** Computed based on the frequency of caption generation.
- **Thread-Safe Logging:**
  - Logging is managed using Python’s `threading.Lock` to ensure safe concurrent updates.
  - Logs are displayed both in the UI and on the Flask log server.

### Thread Safety and Concurrency

- **Video Capture and Processing:**
  - A dedicated thread handles the video capture to avoid UI blocking.
  - The `CaptionThread` processes frames in parallel, emitting signals to update the GUI asynchronously.
- **Flask Server:**
  - Runs on a separate daemon thread, ensuring it does not interfere with the main application.
  - Employs thread-safe mechanisms to share log data with the PyQt5 UI.

---

## Troubleshooting and FAQ

### Common Issues

1. **Video Source Not Starting:**
   - Ensure the selected video file or stream URL is valid.
   - Verify that the required codecs and drivers are installed.

2. **Model Loading Errors:**
   - Check that the Transformers library is installed and updated.
   - Verify GPU availability if using CUDA.

3. **Flask Server Inaccessible:**
   - Ensure that no other service is running on port 5000.
   - Check firewall settings on your system.

### Frequently Asked Questions

- **Q:** What if the video freezes during processing?  
  **A:** This may be due to insufficient system resources. Try reducing the video resolution or running the application on a more powerful machine.

- **Q:** How can I add support for additional video formats?  
  **A:** Modify the file dialog filter in the `open_file` method within `app.py` to include the desired formats.

- **Q:** Can I customize the fine-tuning parameters of the BLIP model?  
  **A:** Yes, refer to the fine-tuning section and adjust the hyperparameters in your training script accordingly.

---

## Repository Structure and Contribution Guidelines

### Repository Layout

The suggested structure of the repository is as follows:

```
RealTime-Drone-Video-Captioning/
├── app.py                 # Main application code
├── requirements.txt       # Python dependencies list
├── README.md              # This comprehensive documentation file
```

### Contribution Guidelines

1. **Code Style:**  
   - Follow PEP8 standards.
   - Include comprehensive comments and docstrings for clarity.

2. **Pull Requests:**  
   - Ensure your branch is up-to-date with the main branch.
   - Provide detailed descriptions of your changes.
   - Include tests for any new features or bug fixes.

3. **Issue Reporting:**  
   - Provide clear steps to reproduce any issues.
   - Attach relevant log files and screenshots.

---

## Performance Metrics and System Requirements

### System Requirements

- **Minimum Requirements:**
  - Python 3.7 or later
  - 4 GB RAM
  - Integrated graphics (for basic usage)

- **Recommended Requirements:**
  - Python 3.8 or later
  - 16 GB RAM
  - NVIDIA GPU with CUDA support for accelerated inference
  - SSD storage for faster data access

### Performance Metrics

- **Video Processing:**
  - The application typically achieves 25–30 FPS on standard hardware.
- **Captioning Inference:**
  - Depending on the hardware, inference FPS can vary significantly.
- **Monitoring:**
  - Real-time metrics are displayed in the UI and can be logged for post-analysis.

---

## Future Enhancements

1. **Model Improvements:**
   - Explore further fine-tuning with larger, more diverse datasets.
   - Integrate state-of-the-art captioning models as they become available.

2. **User Interface:**
   - Implement customizable UI themes.
   - Add support for multi-language captions.

3. **Advanced Logging:**
   - Enhance log server functionality with filtering and search capabilities.
   - Integrate cloud-based logging for remote monitoring.

4. **Extensibility:**
   - Develop plugins for additional video sources (e.g., IP cameras, RTMP streams).
   - Implement automated testing frameworks for continuous integration.

---

## Appendices and References

### Appendix A: Detailed API Reference

- **MainWindow Class:**  
  - Methods: `init_ui()`, `start_video()`, `stop_video()`, `update_frame()`, `update_caption()`
  - Signals: Emitted via the `CaptionThread` to update UI elements.

- **CaptionThread Class:**  
  - Methods: `run()`, `stop()`
  - Usage: Dedicated thread for handling model inference and caption generation.

### Appendix B: External Resources

- **Python:** [https://www.python.org](https://www.python.org)
- **PyQt5:** [https://www.riverbankcomputing.com/software/pyqt/intro](https://www.riverbankcomputing.com/software/pyqt/intro)
- **OpenCV:** [https://opencv.org](https://opencv.org)
- **PyTorch:** [https://pytorch.org](https://pytorch.org)
- **Transformers:** [https://huggingface.co/transformers](https://huggingface.co/transformers)
- **Flask:** [https://flask.palletsprojects.com](https://flask.palletsprojects.com)

### Appendix C: Licensing and Credits

This project is released under the MIT License. Please refer to the [LICENSE](LICENSE) file for more information. Special thanks to the developers and contributors of PyQt5, OpenCV, PyTorch, and Hugging Face for their invaluable libraries.

---

*For any further questions, enhancements, or contributions, please feel free to open an issue or submit a pull request. Your feedback is highly appreciated!*

---

© 2025 RealTime-Drone-Video-Captioning. All rights reserved.
