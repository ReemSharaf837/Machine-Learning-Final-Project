# Machine-Learning-Final-Project
this the link that include the video and the final_voting_model.pkl file because it was to large to upload 
https://drive.google.com/drive/folders/1Owh6XqG8YYs9Hhtlfin2uKPaDSswz92r?usp=drive_link 


Final Machine Learning Project Report

Title: Heart Attack Risk Prediction Using Machine Learning

Student Name: Reem Sharaf
________________________________________
1. Introduction
   
This project presents a complete machine learning pipeline developed to predict the risk of heart attack using health-related survey data. The dataset collected from KAGGLE The primary goal is to demonstrate how machine learning can support public health by identifying individuals with an elevated risk of heart disease based on various personal, behavioral, and medical factors.
________________________________________
2. Dataset Overview
   
•	Source: kaggle 

•	Number of Rows: Approximately 450,000

•	Number of Features: Over 40 columns

•	Target Variable: HadHeartAttack (Binary: Yes/No)

The dataset includes diverse features such as general health perception, physical activity, sleep duration, age, and demographic information. It was selected specifically for its real-world complexity, presence of missing values, class imbalance, and suitability for predictive modeling.
________________________________________
3. Data Preprocessing

A comprehensive preprocessing strategy was applied to ensure the data was suitable for modeling:
•	Columns with more than 50% missing values were removed.
•	Missing values in numerical columns were imputed using the mean.
•	Missing values in categorical columns were imputed using the mode.
•	Duplicate records were identified and removed.
•	Outliers were detected visually using boxplots, particularly in features such as SleepHours and BMI.
Following these steps, all missing and duplicate data issues were resolved, and the dataset was fully cleaned.
________________________________________
4. Exploratory Data Analysis (EDA)
   
An in-depth exploratory analysis was conducted to uncover trends and relationships:
Univariate Analysis:
•	Distribution of AgeCategory, GeneralHealth, SmokerStatus, and the target variable HadHeartAttack was visualized using countplots.
Bivariate Analysis:
•	Relationships between the target variable and key features such as AgeCategory, GeneralHealth, and SmokerStatus were explored using comparative bar plots with 'HadHeartAttack'.
These visualizations provided valuable insights into which features might be most predictive.
________________________________________
5. Feature Engineering and Scaling


•	One-hot encoding was applied to transform categorical variables into numeric format.
•	StandardScaler was used to normalize the numeric features, ensuring uniform scale and enhancing model performance.
•	The target variable was found to be imbalanced. To address this, SMOTE  was applied to the training set, resulting in a balanced class distribution.
________________________________________
6. Model Development and Evaluation

   
The following classification algorithms were developed and tested:
•	Naive Bayes
•	Logistic Regression
•	Random Forest
•	AdaBoost
•	XGBoost
Each model was evaluated using the following metrics on the test set:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	ROC-AUC
•	Confusion Matrix
________________________________________
7. Hyperparameter Tuning
   
Using GridSearchCV, the following models were tuned:
•	Random Forest: Optimized for n_estimators, max_depth, min_samples_split, min_samples_leaf
•	XGBoost: Tuned for n_estimators, max_depth, learning_rate, subsample
These optimizations significantly improved each model’s F1-score and ROC-AUC during training.
________________________________________
8. Model Comparison and Selection
   
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Naive Bayes	82.0%	0.19	0.64	0.29	0.82
Logistic Regression	94.0%	0.47	0.38	0.42	0.85
Random Forest (tuned)	94.0%	0.45	0.45	0.45	0.87
XGBoost (tuned)	94.0%	0.44	0.47	0.46	0.87
Voting Classifier	94.0%	0.46	0.44	0.45	0.88 
A Voting Classifier was built by combining the three top-performing models: Logistic Regression, Random Forest, and XGBoost. It achieved the best overall balance of performance metrics and was therefore selected as the final model.
________________________________________
9. Model Deployment
    
The final ensemble model, scaler, and encoded feature columns were saved using joblib. A lightweight web application was developed using Flask, allowing users to input sample health data and receive a risk prediction. The application includes:
•	app.py (Flask app)
•	index.html (input form within the /templates folder)
________________________________________
 Conclusion
This project demonstrates the practical application of machine learning to a real-world health dataset. Through a thorough end-to-end pipeline, the project achieved strong predictive performance, particularly through the ensemble model (Voting Classifier), which achieved an F1-score of 0.45 and a ROC-AUC of 0.88.
By identifying high-risk individuals using data-driven techniques, this project highlights how machine learning can play a vital role in supporting early intervention and preventive healthcare strategies

