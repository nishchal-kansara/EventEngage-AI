# EventEngage AI - Predict Attendee Behavior & Experience
Event management teams demand precise behavioral data in order to improve attendee experience, optimize event design, and boost participation.  However, analyzing thousands of attendance records, each of which contains demographic information, professional backgrounds, interests, past interactions, and feedback ratings, is time-consuming and prone to inaccuracy.

The challenge is to develop a predictive modeling and analytics system that:
* Forecasts attendee feedback using regression techniques.
* Classifies attendance likelihood using classification algorithms.
* Automates preprocessing so real users can input raw values (not encoded/scaled values).
* Visualizes key trends, such as preferred event types, interest areas, and attendance behavior.
* Enables real-time predictions via a Streamlit web application.

## 1. Project Overview 
EventEngage AI is a end-to-end Machine Learning solution that enables event organizers to study attendee behavior, predict  engagement results, and make decisions based on data.

The system uses a real-world event dataset and supports:
* Regression: To predict feedback ratings.
* Classification: To predict attendance status.
* Streamlit: User-friendly interface for prediction and Interactive charts for analysis.
* Encoders, scalers, and models saved for future usage have been correctly configured and ready to deploy.

This project was developed as part of my "YBAISolution Task" to demonstrate practical machine-learning skills, end-to-end pipeline creation, and real-world deployment.

## 2. Dataset Description  
The dataset contains detailed information about event attendees, including demographics, engagement scores, event history, preferences, and feedback.

### Columns include:
- Demographics: 
  `Age`, `Gender`, `Location_City`, `Location_Country`

- Event Interaction:  
  `Event_Type`, `Session_Attended`, `Attendance_Status`, `Attendance_History`

- Professional Info:  
  `Professional_Industry`, `Years_of_Experience`, `Job_Title`

- Preferences & Behavior:  
  `Preferred_Event_Type`, `Interests`, `Event_Frequency`, `Engagement_Score`

- Transactions:  
  `Ticket_Price`, `Payment_Method`, `Registration_Source`

- Target Variables:  
  - Regression: `Feedback_Rating`  
  - Classification: `Attendance_Status`

1. Dataset Name: `ybai_original.csv`, `ybai_cleaned.csv`  
2. Source: 
- https://www.datayb.com/datasets/dataset-details/datayb_dataset_details_j2z6gunhqn5s2j3/
- https://nishchal-kansara.web.app/dataset/ybai_original.csv

## 3. Project Structure  
```
EventEngage AI/
│
├── dataset/
│   ├── ybai_cleaned.csv        # Preprocessed, cleaned dataset
│   └── ybai_original.csv          # Raw dataset files
│
├── demonstration/
│   ├── charts/        # Charts Images (EDA)
│   └── video.mp4          # Visual Explaination of Project
│
├── models/
│   ├── label_encoders.pkl      # Encoders for categorical features
│   ├── rf_clf.pkl              # Random Forest Classification model
│   ├── rf_reg.pkl              # Random Forest Regression model
│   ├── scaler_clf.pkl          # Scaler used for classification model
│   └── scaler_reg.pkl          # Scaler used for regression model
│
├── README.md
│
├── app.py                      # Main application script (API/UI/Prediction)
│
└── eventengage-ai.ipynb        # Jupyter notebook for EDA, training & analysis
│
└── requirements.txt
```

## 4. Installation & Setup Instructions  

### Step 1: Clone Repository
```
git clone https://github.com/nishchal-kansara/EventEngage-AI.git
cd EventEngage-AI
```

### Step 2: Create Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```
pip install -r requirements.txt
```

## 5. Running the Streamlit Application
Inside the project folder:
```
streamlit run app.py
```

You will see the interface:
* Feedback Rating Prediction
* Attendance Status Prediction
* Interactive Charts

## 6. Running Jupyter Source Files
The notebook `eventengage-ai.ipynb` was created and executed on Kaggle, so no local Jupyter Notebook setup is required.

This notebook contains:
* Data cleaning
* Missing value handling
* Encoding & scaling
* EDA chart generation
* Model training
* Evaluation

## 7. Model Training Summary
### 7.1 Regression models to predict feedback ratings.
During model experimentation, multiple regression algorithms were evaluated to identify the best-performing approach:
* Linear Regression
* Gradient Boosting Regressor
* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

### Best Performing Model: Random Forest Regressor
Reason for Selection:
* Achieved the highest R² score, indicating strong predictive capability.
* Delivered lower MAE and RMSE compared to other models.
* Provided consistent and stable results across validation datasets.
* Demonstrated robustness to feature variability and non-linear relationships.

### 7.2 Classification models to predict attendance Status
Several classification algorithms were evaluated to determine the most effective model for predicting engagement categories:
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier
* LightGBM Classifier

### Best Performing Model: Random Forest Classifier
Reason for Selection:
* Achieved the highest overall accuracy.
* Delivered the best F1-score, indicating strong performance on both precision and recall.
* Provided stable and reliable predictions across validation splits.

## 8. Feature Engineering Performed
A comprehensive preprocessing pipeline was applied to prepare the dataset for both regression and classification models:
* Label Encoding for categorical features
* Standard Scaling for numerical variables
* Event Month Extraction from date fields
* Flattening of Session & Interest Features to convert nested or list-like values into model-friendly formats
* Handling of Multi-Value Columns through splitting, encoding, and aggregation
* Imputation of Missing Demographic Data using appropriate statistical strategies

### Saved Preprocessing Artifacts
To ensure that user inputs follow the same transformations as the training data, the following preprocessing objects were saved:
```
scaler_reg.pkl
scaler_clf.pkl
label_encoders.pkl
```

## 9. Screenshots & Visualization
### 9.1 Exploratory Data Analysis
<p float="left">
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/1.jpg?raw=true" width="30%" />
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/2.jpg?raw=true" width="30%" />
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/3.jpg?raw=true" width="30%" />
</p>
<p float="left">
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/4.jpg?raw=true" width="30%" />
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/5.jpg?raw=true" width="30%" />
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/6.jpg?raw=true" width="30%" />
</p>
<p float="left">
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/7.jpg?raw=true" width="30%" />
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/8.jpg?raw=true" width="30%" />
  <img src="https://github.com/nishchal-kansara/EventEngage-AI/blob/main/demonstration/charts/9.jpg?raw=true" width="30%" />
</p>

### 9.2 Visual Explaination of Project
Watch: https://drive.google.com/file/d/1NgOVoiLFG3mlVYwryW06ZLYjihpqe9Re/view
Download: https://github.com/nishchal-kansara/EventEngage-AI/raw/refs/heads/main/demonstration/video.mp4

## 10. Author
Hi, I'm Nishchal Kansara. I built EventEngage AI to help event teams predict attendee responses and understand what truly drives engagement - in a way that’s simple, accurate, and actionable.

Nishchal Kansara
* Personal Website: https://nishchal-kansara.web.app/
* LinkedIn: https://www.linkedin.com/in/nishchal-kansara/
* Email ID: kansaranishchal@gmail.com
* Github: https://github.com/nishchal-kansara/
