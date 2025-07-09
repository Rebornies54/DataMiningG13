from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           confusion_matrix, precision_recall_fscore_support,
                           roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data
from config import (MODEL_PARAMS, CV_PARAMS, TEST_SIZE, RANDOM_STATE, 
                   TARGET_COLUMN, MODELS_DIR)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(**MODEL_PARAMS['Logistic Regression'], random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(**MODEL_PARAMS['Random Forest'], random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train SVM model with proper probability estimation"""
    try:
        # Create SVM model with probability estimation
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimation
            random_state=42,
            cache_size=1000  # Increase cache size for better performance
        )
        
        # Fit the model
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training SVM model: {str(e)}")
        return None

def perform_cross_validation(model, X, y):
    cv = StratifiedKFold(**CV_PARAMS)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'all_scores': scores.tolist()
    }

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_curve_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc)
        }
    except:
        auc = None
        roc_curve_data = None
    
    cm = confusion_matrix(y_test, y_pred)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'accuracy': float(accuracy),
        'classification_report': report,
        'auc': float(auc) if auc is not None else None,
        'roc_curve': roc_curve_data,
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist(),
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'f1': float(macro_f1),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist()
    }

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'ObesityDataSet_raw_and_data_sinthetic.csv')
    df = load_data(data_path)
    df_processed, label_encoders, scaler = preprocess_data(df)
    
    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE)
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        
        model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        
        metrics_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"{model_name} saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
    
    print("\nTraining completed successfully!") 