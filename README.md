EAMNet-S: Deepfake Detection 🕵️‍♂️🖼️

EAMNet-S is a lightweight, web-based deepfake detection application. It utilizes a custom MobileNetV3 architecture enhanced with an Attention Mechanism to classify images as either real or AI-generated/manipulated (deepfakes).

🚀 Features

Lightweight Architecture: Built on MobileNetV3, making inference fast and efficient.

Attention Mechanism: Focuses on spatial artifacts often left behind by deepfake generation algorithms.

Web Interface: Simple and intuitive frontend (index.html) for users to upload and analyze images.

Ready for Deployment: Includes a render.yaml configuration for seamless deployment on Render.com.

📁 Project Structure

├── app.py                                # Main Python backend (Flask/FastAPI) server
├── index.html                            # Frontend web interface for image uploads
├── deepfake_mobilenetv3_attention.h5     # Pre-trained Keras/TensorFlow model weights
├── requirements.txt                      # Python dependencies (TensorFlow, Flask, etc.)
├── render.yaml                           # Infrastructure-as-code config for Render deployment
└── .python-version                       # Specifies the required Python version


🛠️ Local Setup & Installation

Follow these steps to run the application on your local machine:

1. Clone the repository (or extract the files):

git clone <your-repository-url>
cd EAMNet-S


2. Create a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


3. Install Dependencies:
Ensure you have the required packages installed using requirements.txt:

pip install -r requirements.txt


4. Run the Application:
Start the backend server:

python app.py


The web interface should now be accessible in your browser (typically at http://localhost:5000 or http://localhost:8000).

☁️ Deployment (Render)

This project is pre-configured for easy deployment on Render via the included render.yaml file.

Push this repository to GitHub/GitLab.

Log in to Render.

Go to Blueprints -> New Blueprint Instance.

Connect your repository. Render will automatically read the render.yaml file and deploy the web service and environment precisely as configured.

🧠 Model Details

The underlying model (deepfake_mobilenetv3_attention.h5) leverages a Convolutional Neural Network (CNN) backbone (MobileNetV3-Small) combined with a spatial attention module. This allows the network to prioritize facial anomalies, blending boundaries, and textural inconsistencies that are common in deepfake media, while keeping the computational footprint minimal.
