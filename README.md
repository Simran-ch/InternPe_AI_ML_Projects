# InternPe Internship Repository

This repository contains projects completed during my internship at InternPe. Below is a breakdown of the projects:

## Project 1: Diabetes Prediction with ML
<br>

**Dataset**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
<br>

(In particular, all patients in this dataset are females at least 21 years old of Pima Indian heritage) 



This project is part of my internship at InternPe, where I implemented Support Vector Machine Classifier to predict whether a patient is diabetic or non-diabetic based on various features such as a;
<br>
:) Pregnancies
<br>
:) Glucose
<br>
:) BloodPressure
<br>
:) SkinThickness
<br>
:) Insulin
<br>
:) BMI
<br>
:) DiabetesPedigreeFunction
<br>
:) Age 
<br>

**Project Overview :**
<br>
The goal of this project is to build a classification model that accurately predicts whether a patient is diabetic or non-diabetic based on various factors (as i mentioned above). 
<br>

**Tools Used :**
<br>
**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**Libraries Used** :
<br>
**numpy**: For numerical computations.
<br>
**pandas**: For data manipulation and preprocessing.
<br>
**scikit-learn**: For building and evaluating the linear regression model.
<br>

**Implementation Steps** :
<br>
(1) **Data Preprocessing**: 
<br>
:) Handled missing values, if any.
<br>
:) Performed data exploration.
<br>

(2) **Feature Engineering**:
<br>
**Feature Scaling:** Standardized the dataset to ensure model accuracy and consistency.
<br>

(3) **Model Selection:**
<br>
:) Implemented a Support Vector Machine (SVM) classifier for its efficiency in handling classification problems.
<br>

(4) **Evaluation**:
<br>
:) Used metrics like Accuracy score , Confusion matrix and prepare a detailed classification report to evaluate model performance.
<br>

**Results**:
<br>
:) Achieved an **accuracy score** of 79.15% on training data and 72.08% on test data.
<br>
:) Generated a **confusion matrix** : [[83, 17], [26, 28]].
<br>
:) Produced a detailed **classification report:**
<br>
   **Precision:** 0.76 (class 0), 0.62 (class 1)
<br>
**Recall:** 0.83 (class 0), 0.52 (class 1)
<br>
**F1-Score:** 0.79 (class 0), 0.57 (class 1)
<br>
**Overall Accuracy:** 72% on test data.





## Project 2: CAR PRICE Predictor with ML
<br>


This project is part of my internship at InternPe, where I implemented Linear Regression to estimate vehicle prices based on key features :
<br>
:) name
<br>
:) company
<br>
:) year
<br>
:) kms_driven
<br>
:) fuel_type 

**Project Overview :**
<br>
The goal of this project is to build a CAR price predictor model that accurately predicts the prices based on various factors (as i mentioned above). 
<br>

**Tools Used :**
<br>
**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**Libraries Used** :
<br>
**pandas (pd)** – For data manipulation and analysis
<br>
**numpy (np)** – For numerical computations
<br>
**sklearn.model_selection** – For splitting the dataset (train_test_split)
<br>
**sklearn.preprocessing** – For encoding categorical data (LabelEncoder) and feature scaling (StandardScaler)
<br>
**sklearn.linear_model** – For implementing linear regression (LinearRegression)
<br>
**sklearn.metrics** – For evaluating model performance (mean_absolute_error, mean_squared_error, r2_score)
<br>

**Implementation Steps** :
<br>
(1) **Data Preprocessing**: 
<br>
:) Handled missing values, if any.
<br>
:) Performed data exploration.
<br>

(2) **Feature Engineering**:
<br>
**Feature Scaling:** Standardized the dataset to ensure model accuracy and consistency.
<br>

(3) **Model Selection:**
<br>
:) Implemented Linear Regression.
<br>

(4) **Evaluation**:
<br>
:) Used metrics like Mean absolute error , Mean squared error , Root mean squared error, R2 score to evaluate model performance.
<br>

**Results**:
<br>
:) 𝐌𝐞𝐚𝐧 𝐚𝐛𝐬𝐨𝐥𝐮𝐭𝐞 𝐞𝐫𝐫𝐨𝐫 : 65.48182980256485
<br>
:) 𝐌𝐞𝐚𝐧 𝐬𝐪𝐮𝐚𝐫𝐞𝐝 𝐞𝐫𝐫𝐨𝐫 : 6025.88218179237
<br>
:) 𝐑𝐨𝐨𝐭 𝐦𝐞𝐚𝐧 𝐬𝐪𝐮𝐚𝐫𝐞𝐝 𝐞𝐫𝐫𝐨𝐫 : 77.62655590577474
<br>
:) 𝐑² 𝐒𝐜𝐨𝐫𝐞 : 0.044102324968027684
<br>






## Project 3: IPL WINNING TEAM PREDICTION
<br>

**Dataset** : https://www.kaggle.com/datasets/yuvrajdagur/ipl-dataset-season-2008-to-2017
<br>

This project is part of my internship at InternPe, where i implemented various machine learning models to predict the winning team in an IPL based on key features :
<br>

:) mid: Unique match id.
<br>

:) date: Date on which the match was played.
<br>

:) venue: Stadium where match was played.
<br>

:) batting_team: Batting team name.
<br>

:) bowling_team: Bowling team name.
<br>

:) batsman: Batsman who faced that particular ball.
<br>

:) bowler: Bowler who bowled that particular ball.
<br>

:) runs: Runs scored by team till that point of instance.
<br>

:) wickets: Number of Wickets fallen of the team till that point of instance.
<br>

:) overs: Number of Overs bowled till that point of instance.
<br>

:) runs_last_5: Runs scored in previous 5 overs.
<br>

:) wickets_last_5: Number of Wickets that fell in previous 5 overs.
<br>

:) striker: max(runs scored by striker, runs scored by non-striker).
<br>

:) non-striker: min(runs scored by striker, runs scored by non-striker).
<br>

:) total: Total runs scored by batting team at the end of first innings.
<br>

**Project Overview** :
<br>
This project focuses on predicting the winning team in an IPL (Indian Premier League) match using machine learning models. The dataset used contains historical IPL match data from 2008 to 2017, which has been cleaned, preprocessed, and analyzed to train various classification models for prediction.
<br>


**Tools Used** :
<br>
**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**Libraries Used** :
<br>
**pandas (pd)** – For data manipulation and analysis
<br>
**numpy (np)** – For numerical computations
<br>
**sklearn.model_selection** – For splitting the dataset (train_test_split)
<br>
**sklearn.preprocessing** – For encoding categorical data (LabelEncoder) and feature scaling (StandardScaler)
<br>
**sklearn.linear_model** – For implementing linear regression (LinearRegression)
<br>
**sklearn.metrics** – For evaluating model performance (mean_absolute_error, mean_squared_error, r2_score)
<br>

**Implementation Steps** :
<br>
(1) **Data Preprocessing**:
<br>
:) Removed irrelevant columns
<br>
:) Filtered only consistent teams (teams that played across seasons)
<br>
:) Excluded the first 5 overs of every match (to focus on impactful overs)
<br>
:) Handled missing values and performed feature engineering
<br>

(2) **Feature Encoding** :
<br>
:) Label Encoding for categorical variables
<br>
:) One-Hot Encoding & Column Transformation for better model performance
<br>

(3) **Model Implementation**:
<br>
Trained multiple machine learning models for prediction:
<br>
:) Decision Tree Regressor
<br>
:) Linear Regression
<br>
:) Random Forest Regression
<br>
:) Lasso Regression
<br>
:) Support Vector Machine
<br>
:) Neural Networks
<br>

(4) **Evaluation**:
<br>
:) Used metrics like Mean absolute error , Mean squared error , Root mean squared error to evaluate model performance.

**Results**:
<br>

**Decision Tree Regressor**
<br>
:) Train Score : 99.99%
<br>
:) Test Score : 85.83%
<br>
:) Mean Absolute Error (MAE): 3.980615806532037
<br>
:) Mean Squared Error (MSE): 125.66389304412864
<br>
:) Root Mean Squared Error (RMSE): 11.209990769136638
 <br>

**Linear Regression**
<br>
:) Train Score : 65.99%
<br>
:) Test Score : 65.62%
<br>
:) Mean Absolute Error (MAE): 13.112410435179777
<br>
:) Mean Squared Error (MSE): 305.0563257623772
<br>
:) Root Mean Squared Error (RMSE): 17.465861724013998
<br>

**Random Forest Regression**
<br>
:) Train Score : 99.07%
<br>
:) Test Score : 93.57%
<br>
:) Mean Absolute Error (MAE): 4.398791178425996
<br>
:) Mean Squared Error (MSE): 56.99923941594296
<br>
:) Root Mean Squared Error (RMSE): 7.549784064193026
<br>

**Lasso Regression**
<br>
:) Train Score : 64.95%
<br>
:) Test Score : 64.93%
<br>
:) Mean Absolute Error (MAE): 13.093065567932516
<br>
:) Mean Squared Error (MSE): 311.1276056545018
<br>
:) Root Mean Squared Error (RMSE): 17.638809643921604
 <br>

**Support Vector Machine**
<br>
:) Train Score : 57.41%
<br>
:) Test Score : 57.39%
<br>
:) Mean Absolute Error (MAE): 14.635093385154384
<br>
:) Mean Squared Error (MSE): 378.08184639617673
<br>
:) Root Mean Squared Error (RMSE): 19.444326843482568
<br>

**Neural Networks**
<br>
:) Train Score : 85.45%
<br>
:) Test Score : 85.99%
<br>
:) Mean Absolute Error (MAE): 7.978030236608694
<br>
:) Mean Squared Error (MSE): 124.29835286278254
<br>
:) Root Mean Squared Error (RMSE): 11.148917116150004
<br>

**From above results, we can see that Random Forest performed the best, closely followed by Decision Tree and Neural Networks. So we will be choosing Random Forest for the final model**
<br>








## Project 4: Breast Cancer Classification with a simple Neural Network (NN)
<br>

**--Dataset Overview--**
<br>

**Source:** Scikit-learn Breast Cancer Dataset
<br>
**Features:** Mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
<br>
A total of 30 features extracted from digitized images of breast mass.
<br>
**Target Variable:** 0 (Malignant) and 1 (Benign)
<br>


**--Project Overview--** 
<br>

This project aims to detect breast cancer using deep learning techniques. The dataset used is sourced from Scikit-learn's breast cancer dataset, which includes various medical features to classify whether a tumor is malignant or benign. The model is trained using a Neural Network to assist in early detection and diagnosis.
<br>


**--Tools Used--** 
<br>

**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**--Libraries Used--**
<br>

**pandas (pd)** – For data manipulation and analysis
<br>
**numpy (np)** – For numerical computations
<br>
**Matplotlib** – Data visualization
<br>
**Scikit-learn** – Dataset loading, model training, and evaluation.
<br>


**--Implementation Steps--** 
<br>

(1) **Data Preprocessing**:
<br>

:) Loading the dataset from Scikit-learn
<br>
:) Converting the dataset into a Pandas DataFrame
<br>
:) Checking for missing values and handling them if necessary
<br>
:) Splitting data into features (X) and target variable (Y)
<br>
:) Normalizing feature values for better neural network performance
<br>

(2) **Exploratory Data Analysis (EDA)**:
<br>

:) Checking the distribution of benign and malignant cases
<br>
:) Computing statistical summaries of features
<br>
:) Visualizing data distributions
<br>

(3) **Feature Engineering**:
<br>

:) Selecting relevant features for classification
<br>
:) Normalizing feature values if necessary
<br>

(4) **Model Selection & Training**:
<br>

:) Splitting the dataset into training and testing sets (80-20 ratio)
<br>
:) Implementing a Deep Neural Network (DNN) using TensorFlow/Keras
<br>

(5) **Model architecture**
<br>

**Input Layer** (30 neurons)
<br>
**Hidden Layers** (Dense layers with ReLU activation)
<br>
**Output Layer** (1 neuron with Sigmoid activation)
<br>

(6) **Compiling the model** using Binary Crossentropy Loss and Adam Optimizer
<br> 

(7) **Training the model** with multiple epochs and batch size tuning
<br> 
 
**--Results--**
<br>

**Model Accuracy:** 98.26% (Highly accurate in predicting breast cancer).
<br>
**Prediction Approach:**
<br>
:) Model predicts whether a tumor is Malignant (0) or Benign (1) based on 30 medical features.
<br>
:) Single patient prediction example shows a well-standardized pipeline for real-world usage.
<br>




**Author**: Simran Chaudhary
<br>
Role: Artificial Intelligence and Machine Learning Intern at InternPe
<br>
**LinkedIn**: https://www.linkedin.com/in/simran-chaudhary-5533b7308/
<br>
**GitHub**: https://github.com/Simran-ch
<br>

## Conclusion
<br>
This repository showcases my work on various AI/ML tasks during my internship. Each task focuses on solving specific problems using different machine learning techniques.
