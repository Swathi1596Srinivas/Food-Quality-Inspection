# Food-Quality-Inspection

📌 Project Overview

Food Quality Inspection is an AI-based computer vision project designed to identify and assess the quality of food items using deep learning techniques.

The system uses image processing and machine learning models to detect food items, classify their quality, and provide useful visual results. The main goal is to reduce manual inspection effort and make food quality assessment faster, more consistent, and efficient.

🎯 Objectives

- Detect food items from input images.
- Identify different food/fruit species.
- Classify food quality using deep learning.
- Analyze prediction confidence and model performance.
- Provide an easy-to-use interface for testing the model.

🛠️ Technologies Used

- Python
- YOLOv9
- ResNet-18
- LSTM
- OpenCV
- NumPy
- Pillow
- PyTorch
- Streamlit
- Matplotlib

🔄 Project Workflow

1. Upload or provide an input image.
2. Preprocess the image.
3. Detect the food item using the object detection model.
4. Classify the detected food item based on its quality.
5. Analyze the prediction results.
6. Display the final result through the application interface.

📂 Project Files

File| Description
"app.py"| Main application file
"12.py"| Python implementation file
"train_fruit_yolo.py"| YOLO model training script
"train_species_v2.py"| Species classification training script
"fruit_species_classes.json"| Class/category information
"fruit_species_classifier.pth"| Trained classification model
"confusion_matrix.png"| Model confusion matrix
"metrics_report.txt"| Model performance metrics
"precision_confidence_curve.png"| Precision-confidence curve
"recall_confidence_curve.png"| Recall-confidence curve
"f1_confidence_curve.png"| F1-confidence curve

📊 Model Evaluation

The project includes different evaluation outputs such as:

- Confusion Matrix
- Precision-Confidence Curve
- Recall-Confidence Curve
- F1-Confidence Curve
- Classification Metrics

These results help evaluate the performance and reliability of the trained models.

🚀 How to Run

1. Clone the repository

git clone https://github.com/Swathi1596Srinivas/Food-Quality-Inspection.git

2. Open the project folder

cd Food-Quality-Inspection

3. Install required packages

pip install -r requirements.txt

4. Run the application

streamlit run app.py

💡 Applications

This system can be useful in:

- Food processing industries
- Agriculture
- Quality control
- Freshness inspection
- Automated food sorting
- Research and educational applications

🔮 Future Enhancements

- Support for more food categories.
- Improve model accuracy with larger datasets.
- Real-time camera-based quality inspection.
- Cloud-based deployment.
- Mobile application integration.
- Automated quality reports.

👩‍💻 Author

P.V. Swathi

This project was developed as an AI-based solution for automated food quality inspection using computer vision and deep learning.
