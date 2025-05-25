import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data
from model_training import (train_logistic_regression, train_random_forest, train_svm,
                          evaluate_model, save_model, save_metrics)
from visualization import plot_correlation_matrix, plot_feature_importance, plot_confusion_matrix
from config import TARGET_COLUMN, MODELS_DIR, VISUALIZATIONS_DIR

def get_project_root():
    """
    Get the absolute path to the project root directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def print_evaluation_results(model_name, results, cv_results=None):
    """
    Print evaluation results in a formatted way
    """
    print(f"\n{'='*50}")
    print(f"Results for {model_name}")
    print(f"{'='*50}")
    
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    
    if cv_results:
        print(f"\nCross-validation Results:")
        print(f"Mean CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    
    if results['auc']:
        print(f"\nROC AUC Score: {results['auc']:.4f}")
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    metrics_df = pd.DataFrame({
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1'],
        'Support': results['support']
    })
    print("\nPer-class Metrics:")
    print(metrics_df)

def main():
    # Create necessary directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'ObesityDataSet_raw_and_data_sinthetic.csv')
    df = load_data(data_path)
    df_processed, label_encoders, scaler = preprocess_data(df)
    
    # Split data
    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    models = {
        'Logistic Regression': train_logistic_regression,
        'Random Forest': train_random_forest,
        'SVM': train_svm
    }
    
    for model_name, train_func in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train model
        model = train_func(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        save_model(model, model_path)
        
        # Save metrics
        metrics_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_metrics.json")
        save_metrics(metrics, metrics_path)
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        if metrics['auc'] is not None:
            print(f"AUC: {metrics['auc']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])

if __name__ == "__main__":
    main() 