# Real-estate-price-forecasting-project

Real estate price prediction project (house price prediction using data for california houses)

# California Housing Price Prediction - End-to-End Machine Learning Regression Project

# Overview

This project implements a complete machine learning workflow to predict housing prices using the California Housing dataset.

The objective is to build, evaluate, and optimize regression models using professional ML practices including **pipelines**, **cross-validation**, and hyperparameter tuning.

This project focuses on writing clean, modular, and reproducible machine learning code similar to real-world production workflows.

### Process: EDA → preprocessing → baseline model → model selection (CV) → tuning (GridSearchCV) → final evaluation → inference.


## Project Goals

- Perform structured Exploratory Data Analysis (EDA)

- Build preprocessing pipelines to avoid data leakage

- Compare multiple regression models

- Optimize performance using cross-validation

- Tune hyperparameters using GridSearchCV

- Evaluate final model on unseen test data

- Create a reusable prediction function for inference

## Dataset

- Source: Scikit-Learn California Housing Dataset

- Type: Structured tabular data

- Target Variable: Median House Value

- Features Include:

      - Median Income
      
      - Housing Median Age
      
      - Total Rooms
      
      - Total Bedrooms
      
      - Population
      
      - Households
      
      - Latitude
      
      - Longitude

## Exploratory Data Analysis (EDA)

- Performed detailed EDA including:
    
    - Distribution analysis
    
    - Correlation matrix visualization
    
    - Outlier detection
    
    - Feature relationship analysis
    
    - Target distribution assessment

- Key insight:

    - Median income is the strongest predictor of housing price.

## Machine Learning Pipeline

- To prevent data leakage and ensure reproducibility, preprocessing and modeling were implemented using:

    - Pipeline
    
    - ColumnTransformer

- Preprocessing Steps:

    - Feature scaling for numerical features
    
    - Consistent transformation during training and testing

  This ensures:

    - Clean separation of training and test data
    
    - Proper cross-validation behavior
    
    - Production-ready structure

## Models Evaluated

The following regression models were trained and compared:

    - Linear Regression
    
    - Ridge Regression
    
    - Lasso Regression
    
    - Random Forest Regressor
    
    - HistGradientBoostingRegressor (Best Performer)

## Model Evaluation Strategy

- Used 5-Fold Cross-Validation for reliable performance estimation.

- Evaluation Metrics:

    RMSE (Primary Metric)
    
    MAE
    
    R² Score

-  Cross-validation ensures robust evaluation and reduces overfitting risk.

## Hyperparameter Tuning

- Hyperparameter optimization was performed using:

    - GridSearchCV


- Tuned parameters included:
    
    - Learning rate
    
    - Maximum depth
    
    - Maximum leaf nodes
    
    - L2 regularization
    
    - Number of iterations

This significantly improved model performance compared to default parameters.

## Final Model

Best Model: HistGradientBoostingRegressor
Optimization Method: GridSearchCV
Evaluation: Tested on unseen test dataset

The final model demonstrated strong generalization performance.

## Inference

A reusable prediction function was implemented to:

    - Accept new housing data
    
    - Apply preprocessing automatically
    
    - Generate predicted house values


## Technologies Used

  - Python
  
  - NumPy
  
  - Pandas
  
  - Scikit-Learn
  
  - Matplotlib
  
  - Seaborn

## Future Improvements

- Feature engineering (ratio-based features)

- Log transformation of target variable

- Try XGBoost / LightGBM

- Add model persistence (joblib)

- Build simple API for deployment

- Add experiment tracking

## Key Learnings

- Importance of pipelines to prevent data leakage

- Why cross-validation is critical for reliable evaluation

- How hyperparameter tuning improves performance

- Understanding bias–variance tradeoff

- Structured ML workflow design


## Author

Mahmoud Najdy

Machine Learning Engineer | Data Scientist | Mechatronics Engineer
