import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import (CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TEST_SIZE, 
                   RANDOM_STATE, TARGET_COLUMN, FEATURE_ENGINEERING)

def load_data(file_path):
    """
    Load the obesity dataset from CSV file
    """
    return pd.read_csv(file_path)

def create_engineered_features(df):
    """
    Create additional features through feature engineering
    """
    df_eng = df.copy()
    
    if FEATURE_ENGINEERING['create_bmi']:
        # Calculate BMI
        df_eng['BMI'] = df_eng['Weight'] / (df_eng['Height'] ** 2)
    
    if FEATURE_ENGINEERING['create_age_groups']:
        # Create age groups
        df_eng['Age_Group'] = pd.cut(df_eng['Age'], 
                                   bins=[0, 18, 25, 35, 50, 100],
                                   labels=['0-18', '19-25', '26-35', '36-50', '50+'])
    
    if FEATURE_ENGINEERING['create_activity_score']:
        # Create activity score
        df_eng['Activity_Score'] = df_eng['FAF'] * (1 - df_eng['TUE']/3)
    
    return df_eng

def preprocess_data(df, label_encoders=None, scaler=None, is_training=True):
    """
    Preprocess the data for training or prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    label_encoders : dict, optional
        Dictionary of label encoders for categorical variables
    scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted scaler for numerical variables
    is_training : bool, default=True
        Whether this is training data or prediction data
    
    Returns:
    --------
    tuple
        If is_training=True: (processed_df, label_encoders, scaler)
        If is_training=False: processed_df
    """
    df_processed = df.copy()
    
    if is_training:
        label_encoders = {}
        scaler = StandardScaler()
    
    df_processed = create_engineered_features(df_processed)
    
    if TARGET_COLUMN in df_processed.columns and is_training:
        label_encoders[TARGET_COLUMN] = LabelEncoder()
        df_processed[TARGET_COLUMN] = label_encoders[TARGET_COLUMN].fit_transform(df_processed[TARGET_COLUMN])
    
    for feature in CATEGORICAL_FEATURES:
        if feature in df_processed.columns:
            if is_training:
                label_encoders[feature] = LabelEncoder()
                df_processed[feature] = label_encoders[feature].fit_transform(df_processed[feature])
            else:
                df_processed[feature] = label_encoders[feature].transform(df_processed[feature])
    
    if FEATURE_ENGINEERING['create_age_groups']:
        if is_training:
            label_encoders['Age_Group'] = LabelEncoder()
            df_processed['Age_Group'] = label_encoders['Age_Group'].fit_transform(df_processed['Age_Group'])
        else:
            df_processed['Age_Group'] = label_encoders['Age_Group'].transform(df_processed['Age_Group'])
    
    numerical_features = [f for f in NUMERICAL_FEATURES if f in df_processed.columns]
    if FEATURE_ENGINEERING['create_bmi']:
        numerical_features.append('BMI')
    if FEATURE_ENGINEERING['create_activity_score']:
        numerical_features.append('Activity_Score')
    
    if numerical_features:
        if is_training:
            df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
        else:
            df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
    
    feature_order = (
        CATEGORICAL_FEATURES + 
        NUMERICAL_FEATURES + 
        (['BMI'] if FEATURE_ENGINEERING['create_bmi'] else []) +
        (['Age_Group'] if FEATURE_ENGINEERING['create_age_groups'] else []) +
        (['Activity_Score'] if FEATURE_ENGINEERING['create_activity_score'] else [])
    )
    
    for feature in feature_order:
        if feature not in df_processed.columns:
            df_processed[feature] = 0
    
    if is_training:
        df_processed = df_processed[feature_order + [TARGET_COLUMN]]
    else:
        df_processed = df_processed[feature_order]
    
    return (df_processed, label_encoders, scaler) if is_training else df_processed

def split_data(df, target_column=TARGET_COLUMN):
    """
    Split the data into training and testing sets
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE) 