# real-vs-ai-image-classifier
A deep learning-based web application that detects whether an image is AI-generated or original (real) using a fine-tuned ResNet18 model.

🚀 Project Overview

With the rapid rise of AI-generated content, especially in the form of hyper-realistic images, there's a growing need to detect and differentiate between real and AI-generated visuals. This project leverages the ResNet18 architecture to classify images and provides a user-friendly interface for quick and effective predictions.

🛠 Features

📂 Upload any image for classification
🧠 Deep learning powered (PyTorch + ResNet18)
🌐 Simple and fast web app using Streamlit
🧪 Binary classification: AI-Generated or Original
📁 Repository Structure

.
├── ai_detector_app.py        # Streamlit app script
├── mt.py                     # Model utility functions (loading, preprocessing, prediction)
├── resnet18-ai-detector.pth  # Trained model weights
├── README.md                 # You're here!
🔧 Setup Instructions

1. Clone the repository
git clone https://github.com/your-username/ai-image-detector.git
cd ai-image-detector
2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
If requirements.txt doesn't exist yet, here's a basic list you can use:

torch
torchvision
streamlit
Pillow
4. Run the Streamlit app
streamlit run ai_detector_app.py

🧠 Model Details

Architecture: ResNet18
Framework: PyTorch
Task: Binary Classification – Detect if the image is AI-generated or Original
Trained on: Dataset of real vs AI-generated images (custom)
Model weights are stored in resnet18-ai-detector.pth.

📷 Example


Input Image	Prediction
Original
AI-Generated

📌 To-Do

 Add more diverse datasets for improved generalization
 Deploy on Hugging Face or Streamlit Cloud
 Improve model accuracy using ensemble learning

🙌 Acknowledgments

PyTorch
Streamlit
Community datasets for real and AI-generated images
