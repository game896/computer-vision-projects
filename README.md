# Computer Vision Practice Projects

This repository contains three practical Computer Vision tasks developed as exercises in image processing, object detection, and character recognition.

---

## Task 1: Coin Value Counting

**Description:**  
Detection and calculation of the total value of coins on images. Gold coins are valued at 1, star coins at 5.

**Approach:**  
- Image preprocessing (grayscale conversion, contrast enhancement, median blur)
- Circle detection using **Hough Circle Transform** for two different coin sizes
- Position-based filtering of detected circles
- Total value calculation per image

**Example Input:**

![Input 1](res/k1.jpg)  

---

## Task 2: Blue Bones Counting in Videos

**Description:**  
Counting the number of blue bones appearing in video sequences.

**Approach:**  
- Frame-by-frame video processing
- Motion detection using frame differencing and Gaussian blur
- Definition of Regions of Interest (ROI) in upper and lower frame parts
- Circle detection using **Hough Circle Transform**
- State tracking to prevent duplicate counting

**Example Input:**

![Input 1](res/k2.gif)  

---

## Task 3: Character Recognition on Images

**Description:**  
Recognition of letter sequences from images using a neural network.

**Approach:**  
- Programmatic creation of training dataset from selected images
- Image preprocessing (grayscale, binarization, erosion)
- Automatic deskewing using Hough Line Transform
- Contour detection and region extraction
- Training a simple **Artificial Neural Network (ANN)** with TensorFlow/Keras
- Prediction and Hamming distance calculation

**Example Input:**

![Input 1](res/k3.png)  

---

## Technologies Used

- **Python 3**
- **OpenCV** — main computer vision library
- **NumPy** — numerical computations
- **TensorFlow / Keras** — neural network model
- **Pandas** — ground truth data handling
- **SciPy (ndimage)** — image rotation

---
