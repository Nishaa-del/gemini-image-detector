
# Fake Image Detection Using SVM

This project is a machine learning–based web application that detects whether an uploaded image is **Real** or **AI-generated (Gemini-generated)**.
It uses a **Support Vector Machine (SVM)** classifier trained on image features and provides a **Streamlit-based web interface** for image upload and prediction.



## Project Overview

With the rapid growth of AI image generators such as Google Gemini, DALL·E, and Midjourney, identifying AI-generated images has become an important challenge.
This project addresses this problem by implementing a fake image detection system using classical machine learning techniques.



## Objectives

* Detect whether an image is Real or Gemini-generated
* Train an SVM classifier using image pixel features
* Provide a simple web interface for image upload and prediction
* Demonstrate AI image detection for academic purposes



## Technologies Used

* Programming Language: Python
* Machine Learning Algorithm: Support Vector Machine (SVM)
* Libraries:

  * scikit-learn
  * NumPy
  * Pillow (PIL)
  * Streamlit
  * Joblib
* Development Environment: VS Code



## Project Structure

```
gemini_detector/
│
├── dataset/
│   ├── real/          # Real images
│   └── gemini/        # Gemini-generated images
│
├── svm_detector.py    # Model training script
├── gemini_web_app.py  # Streamlit web application
├── svm_model.pkl      # Trained model file
├── README.md          # Project documentation
└── requirements.txt   # Required dependencies
```


## Working Methodology

1. Images are resized and converted into numerical feature vectors
2. An SVM classifier is trained to distinguish between real and Gemini-generated images
3. The trained model is saved using Joblib
4. A Streamlit web application allows users to upload an image
5. The system predicts the image category and displays the result


## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/gemini-image-detector.git
cd gemini-image-detector
```

### Step 2: Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python svm_detector.py
```

### Step 5: Run the Web Application

```bash
streamlit run gemini_web_app.py
```

The application will open in the browser at:

```
http://localhost:8501
```


## Sample Output

* Real Image: Classified as real
* Gemini-generated Image: Classified as AI-generated
<img width="1160" height="602" alt="image" src="https://github.com/user-attachments/assets/c5e092b7-054d-4134-9c95-80d37261e1d7" />


<img width="1157" height="603" alt="image" src="https://github.com/user-attachments/assets/9e5d2952-4e62-4d8c-8875-266e67862f0d" />


<img width="1146" height="605" alt="image" src="https://github.com/user-attachments/assets/da9b243a-3c07-4589-b702-a4b9c97d6172" />





## Limitations

* Uses a synthetic dataset for demonstration
* Model accuracy depends on dataset quality
* Not trained on official Gemini image datasets



## Future Enhancements

* Integrate deep learning models such as CNNs
* Use large-scale real-world AI-generated datasets
* Extend detection to video deepfakes
* Deploy the application on cloud platforms



## References

* Scikit-learn Documentation
* Streamlit Documentation
* Research papers on AI-generated image detection



## Author

Nisha A
Department of Computer Science and Engineering
AIML Mini Project



