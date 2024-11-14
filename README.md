# Team-4-Repository

# AI-Driven Grocery Management System

## Overview

Welcome to the **AI-Driven Grocery Management System**! This project aims to automate the process of tracking household inventory and recommending custom recipes based on the available ingredients. It combines cutting-edge AI technologies for receipt text localization, inventory management, and recipe recommendation to create a seamless user experience. The application is built to run on both desktop and mobile devices using the Kivy framework.

## Project Structure

The project is organized into the following sections:

1. **Text Localization and Recipe Recommender**  
   This section contains the models and scripts responsible for text localization (OCR) and recipe recommendations.  
   - **Receipt Text Localization**: Implements deep learning models and compares them for detecting bounding boxes in receipts.

2. **Recipe Recommender**
   - **Recipe Recommendation**: Leverages a fine-tuned GPT-2 model trained on almost 230,000 recipes to recommend personalized recipes based on the available ingredients and cooking time.

2. **Grocery App**  
   This is the core of the app, built with Kivy and Flask. It includes:
   - **Welcome Screens**: A home screen for navigating to the camera, inventory and generate recipe screens.
   - **Receipt Capture**: Uses Canny Edge detection, Hough transform, MobileNet and FasterRCNN to capture receipts using a camera and extract the items purchased.
   - **Inventory Management**: Tracks the user’s groceries and recommends recipes accordingly.
   - **Recipe Recommender**: Recommends recipes based on user's inputs of ingredients (Chosen from inventory) and cooking time.

---

## Features

- **Receipt Capture and OCR**:  
  Uses **FasterRCNN** with **MobileNet** to localize and extract text from receipts. Only done after using Contours and Hough transform to find bounding box around the whole receipt.
  
- **Custom Recipe Recommendations**:  
  Based on the detected ingredients from the receipt, the system recommends recipes tailored to the user’s available inventory and cooking time using a fine-tuned **GPT-2 model**.

- **Inventory Tracking**:  
  Automatically updates inventory and allows users to keep track of what they have at home, suggesting recipes based on available ingredients.

- **Cross-Platform**:  
  Built using **Kivy**, the app is compatible with both desktop and mobile platforms, with integration of **TensorFlow Lite** for lightweight mobile inference.

---

## Technologies Used

- **Programming Languages**: Python
- **Deep Learning Frameworks**: PyTorch, TensorFlow, Keras
- **Libraries**:
  - OpenCV (for image processing)
  - Kivy (for app development)
  - Matplotlib (for evaluation)
  - Flask (for local server integration)
- **Pretrained Models**:
  - **FasterRCNN with MobileNet backbone** (for text localization)
  - **GPT-2** (for recipe recommendation)
  - **MobileNet** (for bounding box prediction)
- **Other Tools**:
  - Pandas (for data management)
  - Numpy (for numerical operations)
