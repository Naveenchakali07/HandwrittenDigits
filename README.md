# 🏡 House Price Prediction (PRCP-1020)

## 📌 Project Overview
This project focuses on predicting house prices based on various features such as number of rooms, location, area, and other property-related attributes. The goal is to analyze the dataset, preprocess the data, build machine learning models, and evaluate their performance to provide reliable price predictions.

## 🎯 Objectives
- Perform data cleaning and preprocessing on the housing dataset.
- Conduct Exploratory Data Analysis (EDA) to identify patterns and insights.
- Apply multiple machine learning algorithms for price prediction.
- Compare models based on accuracy and error metrics.
- Visualize results for better interpretability.

## 📊 Dataset
- **Source:** Provided dataset (`data.csv`)
- **Features:** Includes property attributes such as rooms, bathrooms, location, area, etc.
- **Target Variable:** `Price`

## 🛠️ Technologies Used
- Python 🐍
- Pandas, NumPy – Data handling
- Matplotlib, Seaborn – Data visualization
- Scikit-learn – Machine Learning models

## 🚀 Workflow
1. **Data Loading & Inspection**
2. **Data Cleaning & Preprocessing**
   - Handling missing values  
   - Encoding categorical variables  
   - Scaling numerical features  
3. **Exploratory Data Analysis**
   - Distribution plots  
   - Correlation heatmaps  
   - Feature importance  
4. **Model Development**
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - (Others tested in notebook)  
5. **Model Evaluation**
   - R² Score  
   - RMSE (Root Mean Square Error)  
   - Comparison of results  
6. **Visualization**
   - Predicted vs Actual prices  
   - Feature impact on house price  

## 📈 Results
- The **Random Forest Regressor** provided the best performance with higher accuracy and lower RMSE compared to other models.
- Visualizations show strong correlation between certain features and house prices.

## 📌 Future Improvements
- Implement advanced models like XGBoost or Gradient Boosting.
- Hyperparameter tuning for better performance.
- Deploy the model using Flask/Django for real-world applications.

## 🙌 Acknowledgments
This project was developed as part of the **Capstone Project (PRCP-1020)** for Data Science learning.

---
