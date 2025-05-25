from xml.parsers.expat import model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from data_preprocessing import load_data, preprocess_data
from config import TARGET_COLUMN, MODELS_DIR, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from custom_random_forest import CustomRandomForest
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Obesity Level Prediction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stDataFrame {
        background-color: white;
        padding: 1rem;
        border-radius: 0.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def format_number(x):
    """Format numbers for display"""
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    elif isinstance(x, (float, np.floating)):
        return f"{x:.2f}"
    return x

def load_results():
    """Load all saved models"""
    models = {}
    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith('.pkl'):
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            model_path = os.path.join(MODELS_DIR, model_file)
            models[model_name] = joblib.load(model_path)
    return models

def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    return plt

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return None
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance', fontsize=14, pad=20)
    bars = plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def plot_roc_curve(fpr, tpr, auc):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='#2c3e50', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    return plt

def display_model_metrics(metrics, label_encoders):
    """Display model metrics in a structured format"""
    # Overall metrics
    st.markdown("### Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1']:.4f}")
    
    # Per-class metrics
    st.markdown("### Per-Class Performance")
    class_metrics = pd.DataFrame({
        'Class': label_encoders[TARGET_COLUMN].classes_,
        'Precision': [f"{x:.4f}" for x in metrics['precision_per_class']],
        'Recall': [f"{x:.4f}" for x in metrics['recall_per_class']],
        'F1-Score': [f"{x:.4f}" for x in metrics['f1_per_class']],
        'Support': [format_number(x) for x in metrics['support_per_class']]
    })
    
    # Reset index to ensure unique indices
    class_metrics = class_metrics.reset_index(drop=True)
    
    st.dataframe(class_metrics.style.background_gradient(cmap='YlOrRd', subset=['Precision', 'Recall', 'F1-Score']), 
                use_container_width=True)
    
    st.markdown("### Detailed Classification Report")
    st.text(metrics['classification_report'])

def plot_model_comparison(metrics_dict, metric_name):
    """Plot comparison of a specific metric across models"""
    plt.figure(figsize=(10, 6))
    models = list(metrics_dict.keys())
    values = [metrics_dict[model][metric_name] for model in models]
    
    bars = plt.bar(models, values, color=['#2c3e50', '#3498db', '#e74c3c'])
    plt.title(f'{metric_name} Comparison Across Models', fontsize=14, pad=20)
    plt.xticks(rotation=45)
    plt.ylabel(metric_name)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def plot_confusion_matrix_comparison(metrics_dict, label_encoders):
    """Plot confusion matrices for all models side by side"""
    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, len(metrics_dict), figsize=(20, 6))
    
    # If only one model, convert axes to array for consistent indexing
    if len(metrics_dict) == 1:
        axes = np.array([axes])
    
    # Plot each model's confusion matrix
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoders[TARGET_COLUMN].classes_,
                   yticklabels=label_encoders[TARGET_COLUMN].classes_,
                   ax=axes[idx])
        axes[idx].set_title(f'{model_name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    
    plt.tight_layout()
    return plt

def create_input_form():
    """Create input form for prediction"""
    st.subheader("Enter Patient Information")
    
    # Algorithm selection
    st.markdown("### Select Algorithm")
    selected_model = st.selectbox(
        "Choose the algorithm for prediction",
        ["Logistic Regression", "Random Forest", "SVM", "Custom Random Forest"],
        help="Select the algorithm you want to use for prediction"
    )
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # Initialize input dictionary
    input_data = {}
    
    with col1:
        # Numerical inputs
        st.markdown("### Physical Measurements")
        input_data['Age'] = float(st.number_input("Age", min_value=0, max_value=100, value=30))
        input_data['Height'] = float(st.number_input("Height (meters)", min_value=0.0, max_value=3.0, value=1.7, format="%.2f"))
        input_data['Weight'] = float(st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, format="%.1f"))
        
        st.markdown("### Eating Habits")
        input_data['FCVC'] = float(st.slider("Frequency of consumption of vegetables (1-3)", 1, 3, 2))
        input_data['NCP'] = float(st.slider("Number of main meals (1-4)", 1, 4, 3))
        input_data['CH2O'] = float(st.slider("Daily water consumption (1-3)", 1, 3, 2))
        
        st.markdown("### Physical Activity")
        input_data['FAF'] = float(st.slider("Physical activity frequency (0-3)", 0, 3, 1))
        input_data['TUE'] = float(st.slider("Time using technology devices (0-2)", 0, 2, 1))
    
    with col2:
        # Categorical inputs
        st.markdown("### Personal Information")
        input_data['Gender'] = st.selectbox("Gender", ["Female", "Male"])
        input_data['family_history_with_overweight'] = st.selectbox("Family History with Overweight", ["yes", "no"])
        
        st.markdown("### Eating Behavior")
        input_data['FAVC'] = st.selectbox("Frequent consumption of high caloric food", ["yes", "no"])
        input_data['CAEC'] = st.selectbox("Consumption of food between meals", 
                                         ["no", "Sometimes", "Frequently", "Always"])
        input_data['SMOKE'] = st.selectbox("Smoking", ["yes", "no"])
        input_data['SCC'] = st.selectbox("Calories consumption monitoring", ["yes", "no"])
        input_data['CALC'] = st.selectbox("Alcohol consumption", 
                                         ["no", "Sometimes", "Frequently", "Always"])
        input_data['MTRANS'] = st.selectbox("Transportation used", 
                                          ["Public_Transportation", "Automobile", "Motorbike", "Bike", "Walking"])
    
    return input_data, selected_model

def train_custom_random_forest(X_train, y_train):
    """Train the custom Random Forest model with optimized parameters"""
    try:
        # Convert to numpy arrays if they aren't already
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Ensure data is properly shaped
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(1, -1)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Training progress: {int(progress * 100)}%")
        

        model = CustomRandomForest(
            n_estimators=100,      
            max_depth=None,        
            min_samples_split=2,   
            min_samples_leaf=1,    
            max_features='sqrt',   
            random_state=42,
            bootstrap=True,       
            oob_score=False,       
            class_weight='balanced' 
        )
        
        # Train the model with progress tracking
        model.fit(X_train, y_train, progress_callback=update_progress)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return model
    except Exception as e:
        st.error(f"Error training custom Random Forest: {str(e)}")
        st.error("Please check the data format and try again.")
        return None

def display_prediction(prediction, probabilities, model_name, label_encoders, confidence_threshold=0.6):
    """Display prediction results with confidence assessment"""
    if prediction is None or probabilities is None:
        return
        
    st.subheader("Prediction Results")
    
    # Display prediction
    st.markdown(f"### {model_name} Prediction")
    
    # Get the highest probability
    max_prob = max(probabilities.values())
    prediction_confidence = f"{max_prob:.1%}"
    
    # Determine confidence level
    if max_prob >= confidence_threshold:
        confidence_level = "High"
        confidence_color = "green"
    elif max_prob >= 0.4:
        confidence_level = "Medium"
        confidence_color = "orange"
    else:
        confidence_level = "Low"
        confidence_color = "red"
    
    # Display prediction with confidence
    st.markdown(f"""
    **Predicted Obesity Level:** {prediction}
    
    **Confidence Level:** <span style='color:{confidence_color}'>{confidence_level}</span>
    **Prediction Confidence:** {prediction_confidence}
    """, unsafe_allow_html=True)
    
    # Create DataFrame with raw probabilities
    prob_df = pd.DataFrame({
        'Obesity Level': list(probabilities.keys()),
        'Probability': [float(p) for p in probabilities.values()]  # Ensure float values
    })
    
    # Add formatted percentage column for display
    prob_df['Probability (%)'] = prob_df['Probability'].apply(lambda x: f"{x:.1%}")
    
    # Sort by probability
    prob_df = prob_df.sort_values('Probability', ascending=False)
    
    # Display probability distribution table
    st.markdown("**Probability Distribution:**")
    st.dataframe(
        prob_df[['Obesity Level', 'Probability (%)']].style.background_gradient(
            cmap='YlOrRd', 
            subset=['Probability (%)']
        ),
        use_container_width=True
    )
    
    # Plot probability distribution using raw float values
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(prob_df['Obesity Level'], prob_df['Probability'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{model_name} Probability Distribution')
    plt.ylabel('Probability')
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate labels
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("### Interpretation")
    if confidence_level == "High":
        st.markdown("""
        The model shows high confidence in this prediction, suggesting it's likely to be accurate.
        However, please consider:
        - The prediction is based on the provided information
        - Individual variations may affect the actual outcome
        - Regular health check-ups are recommended
        """)
    elif confidence_level == "Medium":
        st.markdown("""
        The model shows moderate confidence in this prediction.
        Consider:
        - The prediction might be less certain due to:
            - Unusual combinations of factors
            - Borderline cases
            - Limited information in certain areas
        - Additional factors not captured in the model may be relevant
        - Professional medical advice is recommended
        """)
    else:
        st.markdown("""
        The model shows low confidence in this prediction.
        This could be due to:
        - Unusual or extreme values in the input
        - Rare combinations of factors
        - Insufficient information
        
        **Recommendation:** Please consult with a healthcare professional for a more accurate assessment.
        """)

def get_prediction(input_data, model, label_encoders, scaler, confidence_threshold=0.6):
    """Get prediction from the selected model with confidence assessment"""
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Calculate BMI
        height_m = float(input_data['Height'])
        weight_kg = float(input_data['Weight'])
        bmi = weight_kg / (height_m ** 2)
        
        # Validate input ranges
        if height_m < 0.5 or height_m > 3.0:  # Reasonable height range in meters
            st.warning(f"Warning: Height ({height_m:.2f}m) is outside reasonable range (0.5m - 3.0m).")
        if weight_kg < 20 or weight_kg > 300:  # Reasonable weight range in kg
            st.warning(f"Warning: Weight ({weight_kg:.1f}kg) is outside reasonable range (20kg - 300kg).")
        
        # Validate BMI range
        if bmi < 10 or bmi > 100:  # Unreasonable BMI range
            st.warning(f"Warning: Calculated BMI ({bmi:.1f}) is outside reasonable range (10-100). Please check height and weight values.")
        
        # BMI-based validation
        bmi_categories = {
            'Insufficient_Weight': (0, 18.5),
            'Normal_Weight': (18.5, 25),
            'Overweight_Level_I': (25, 30),
            'Overweight_Level_II': (30, 35),
            'Obesity_Type_I': (35, 40),
            'Obesity_Type_II': (40, 50),
            'Obesity_Type_III': (50, float('inf'))
        }
        
        # Find expected category based on BMI
        expected_category = None
        for category, (min_bmi, max_bmi) in bmi_categories.items():
            if min_bmi <= bmi < max_bmi:
                expected_category = category
                break
        
        # Ensure all required features are present
        required_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        # Preprocess the input data
        input_processed = preprocess_data(input_df, label_encoders, scaler, is_training=False)
        
        # Ensure input data is properly shaped
        if len(input_processed.shape) == 1:
            input_processed = input_processed.reshape(1, -1)
        
        # Get prediction
        pred = model.predict(input_processed)[0]
        pred_proba = model.predict_proba(input_processed)[0]
        
        # Convert prediction to original label
        original_label = label_encoders[TARGET_COLUMN].inverse_transform([pred])[0]
        
        # Create probability dictionary
        probabilities = dict(zip(label_encoders[TARGET_COLUMN].classes_, pred_proba))
        
        # Add BMI-based validation warning if prediction differs from BMI category
        if expected_category and original_label != expected_category:
            st.warning(f"""
            Note: The model prediction ({original_label}) differs from the expected category based on BMI ({expected_category}).
            This could be due to other factors like:
            - Eating habits and meal frequency
            - Physical activity levels
            - Family history of overweight
            - Technology usage and sedentary behavior
            """)
        
        # Display BMI information
        st.info(f"""
        **BMI Information:**
        - Calculated BMI: {bmi:.1f}
        - BMI Category: {expected_category if expected_category else 'Unknown'}
        - Model Prediction: {original_label}
        
        **Key Factors Considered:**
        - Physical measurements (Height, Weight, BMI)
        - Eating habits (Meal frequency, vegetable consumption, etc.)
        - Physical activity and lifestyle
        - Family history and personal factors
        """)
        
        return original_label, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("""
        Please check:
        1. All required features are provided
        2. Input values are within reasonable ranges
        3. Data format is correct
        """)
        return None, None

def plot_model_comparison_metrics(custom_metrics, library_metrics, metric_name):
    """Plot comparison of metrics between custom and library implementations"""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Custom RF', 'Library RF']
    values = [custom_metrics[metric_name], library_metrics[metric_name]]
    
    bars = ax.bar(models, values, color=['#2c3e50', '#3498db'])
    ax.set_title(f'{metric_name} Comparison', fontsize=14, pad=20)
    ax.set_ylim(0, 1.1)  # Set y-axis limit to accommodate labels
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_feature_importance_comparison(custom_importance, library_importance, feature_names):
    """Plot feature importance comparison between custom and library implementations"""
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Custom RF': custom_importance,
        'Library RF': library_importance
    })
    
    # Sort by average importance
    comparison_df['Average'] = (comparison_df['Custom RF'] + comparison_df['Library RF']) / 2
    comparison_df = comparison_df.sort_values('Average', ascending=False).head(10)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df['Feature']))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Custom RF'], width, label='Custom RF', color='#2c3e50')
    ax.bar(x + width/2, comparison_df['Library RF'], width, label='Library RF', color='#3498db')
    
    ax.set_ylabel('Importance')
    ax.set_title('Top 10 Feature Importance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig

def display_model_comparison(custom_metrics, library_metrics, feature_names, custom_model, library_model, X_test, y_test, label_encoders):
    """Display comparison between Custom Random Forest and Random Forest"""
    st.header("Model Comparison")
    
    # Create tabs for different comparison views
    comparison_tab1, comparison_tab2, comparison_tab3 = st.tabs([
        "Performance Metrics", "Feature Importance", "Confusion Matrices"
    ])
    
    with comparison_tab1:
        st.subheader("Performance Metrics Comparison")
        
        # Create comparison DataFrame with updated model names
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
            'Value': [
                custom_metrics['accuracy'],
                custom_metrics['precision'],
                custom_metrics['recall'],
                custom_metrics['f1'],
                library_metrics['accuracy'],
                library_metrics['precision'],
                library_metrics['recall'],
                library_metrics['f1']
            ],
            'Model': ['Custom Random Forest'] * 4 + ['Random Forest'] * 4
        })
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model', ax=ax)
        plt.title('Performance Metrics Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display detailed metrics table
        st.subheader("Detailed Metrics Table")
        pivot_df = metrics_df.pivot(index='Metric', columns='Model', values='Value')
        pivot_df['Difference'] = pivot_df['Custom Random Forest'] - pivot_df['Random Forest']
        st.dataframe(pivot_df.style.background_gradient(
            cmap='RdYlGn',
            subset=['Difference'],
            vmin=-0.1,
            vmax=0.1
        ))
        
        # Add explanation of the performance difference
        st.markdown("""
        ### Performance Analysis
        The Random Forest model shows higher performance metrics because:
        1. It uses optimized hyperparameters and implementation
        2. It has better handling of feature interactions
        3. It employs more sophisticated tree building algorithms
        
        The Custom Random Forest implementation, while functional, is a simplified version that:
        1. Uses basic tree building algorithms
        2. Has simpler feature selection methods
        3. May not handle edge cases as effectively
        """)
    
    with comparison_tab2:
        st.subheader("Feature Importance Comparison")
        
        # Get feature importance for both models with updated names
        custom_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': custom_model.feature_importances_,
            'Model': 'Custom Random Forest'
        })
        
        library_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': library_model.feature_importances_,
            'Model': 'Random Forest'
        })
        
        # Combine and sort by average importance
        combined_importance = pd.concat([custom_importance, library_importance])
        top_features = combined_importance.groupby('Feature')['Importance'].mean().nlargest(10).index
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=combined_importance[combined_importance['Feature'].isin(top_features)],
            x='Importance',
            y='Feature',
            hue='Model',
            ax=ax
        )
        plt.title('Top 10 Feature Importance Comparison')
        plt.tight_layout()
        st.pyplot(fig)
    
    with comparison_tab3:
        st.subheader("Confusion Matrices Comparison")
        
        # Calculate confusion matrices
        custom_pred = custom_model.predict(X_test)
        library_pred = library_model.predict(X_test)
        
        custom_cm = confusion_matrix(y_test, custom_pred)
        library_cm = confusion_matrix(y_test, library_pred)
        
        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Custom Random Forest confusion matrix
        sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoders['NObeyesdad'].classes_,
                   yticklabels=label_encoders['NObeyesdad'].classes_,
                   ax=ax1)
        ax1.set_title('Custom Random Forest')
        
        # Random Forest confusion matrix
        sns.heatmap(library_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoders['NObeyesdad'].classes_,
                   yticklabels=label_encoders['NObeyesdad'].classes_,
                   ax=ax2)
        ax2.set_title('Random Forest')
        
        plt.tight_layout()
        st.pyplot(fig)

def initialize_custom_rf(df_processed, label_encoders, scaler):
    """Initialize and train the Custom Random Forest model if not already in session state"""
    # Initialize session state variables if they don't exist
    if 'custom_rf_model' not in st.session_state:
        st.session_state.custom_rf_model = None
    if 'custom_rf_training_data' not in st.session_state:
        st.session_state.custom_rf_training_data = None
    if 'custom_rf_metrics' not in st.session_state:
        st.session_state.custom_rf_metrics = None
    if 'library_rf_metrics' not in st.session_state:
        st.session_state.library_rf_metrics = None

    # Only train if model doesn't exist or training data is missing
    if st.session_state.custom_rf_model is None or st.session_state.custom_rf_training_data is None:
        with st.spinner("Training Custom Random Forest model (this only happens once per session)..."):
            try:
                # Calculate BMI for all samples
                df_with_bmi = df_processed.copy()
                df_with_bmi['BMI'] = df_with_bmi['Weight'] / (df_with_bmi['Height'] ** 2)
                
                # Add BMI to numerical features
                numerical_features = NUMERICAL_FEATURES + ['BMI']
                
                # Split data
                X = df_with_bmi.drop('NObeyesdad', axis=1)
                y = df_with_bmi['NObeyesdad']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
                
                # Train custom model
                custom_rf = train_custom_random_forest(X_train, y_train)
                if custom_rf is None:
                    st.error("Failed to train the Custom Random Forest model")
                    return None
                
                # Get library model for comparison
                library_rf = models['Random Forest']
                
                # Store model and training data in session state
                st.session_state.custom_rf_model = custom_rf
                st.session_state.custom_rf_training_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'feature_names': X_train.columns.tolist()  # Store feature names including BMI
                }
                
                # Calculate predictions for both models
                custom_pred = custom_rf.predict(X_test)
                library_pred = library_rf.predict(X_test)
                
                # Store predictions in session state
                st.session_state.custom_predictions = custom_pred
                st.session_state.library_predictions = library_pred
                
                # Calculate and store metrics
                st.session_state.custom_rf_metrics = {
                    'accuracy': accuracy_score(y_test, custom_pred),
                    'precision': precision_score(y_test, custom_pred, average='weighted'),
                    'recall': recall_score(y_test, custom_pred, average='weighted'),
                    'f1': f1_score(y_test, custom_pred, average='weighted'),
                    'precision_per_class': precision_score(y_test, custom_pred, average=None),
                    'recall_per_class': recall_score(y_test, custom_pred, average=None),
                    'f1_per_class': f1_score(y_test, custom_pred, average=None),
                    'support_per_class': np.bincount(y_test),
                    'classification_report': classification_report(y_test, custom_pred, target_names=label_encoders[TARGET_COLUMN].classes_)
                }
                
                st.session_state.library_rf_metrics = {
                    'accuracy': accuracy_score(y_test, library_pred),
                    'precision': precision_score(y_test, library_pred, average='weighted'),
                    'recall': recall_score(y_test, library_pred, average='weighted'),
                    'f1': f1_score(y_test, library_pred, average='weighted'),
                    'precision_per_class': precision_score(y_test, library_pred, average=None),
                    'recall_per_class': recall_score(y_test, library_pred, average=None),
                    'f1_per_class': f1_score(y_test, library_pred, average=None),
                    'support_per_class': np.bincount(y_test),
                    'classification_report': classification_report(y_test, library_pred, target_names=label_encoders[TARGET_COLUMN].classes_)
                }
                
                st.success("Custom Random Forest model trained successfully!")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.error("Please check the data preprocessing steps.")
                return None
    
    return st.session_state.custom_rf_model

def main():
    st.title("Obesity Level Prediction Analysis Dashboard")
    
    try:
        # Load data and models
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'ObesityDataSet_raw_and_data_sinthetic.csv')
        df = load_data(data_path)
        df_processed, label_encoders, scaler = preprocess_data(df)
        models = load_results()  # This loads all saved models
        
        # Navigation
        st.sidebar.title("Navigation")
        st.sidebar.markdown("---")
        page = st.sidebar.radio("Select Page", ["Data Overview", "Model Performance", "Feature Analysis", "Model Comparison", "Make Prediction", "Custom Random Forest", "RF Comparison"])
        
        if page == "Data Overview":
            st.header("Dataset Overview")
            
            # Basic statistics
            st.subheader("Basic Statistics")
            stats_df = df.describe()
            formatted_stats = stats_df.applymap(format_number)
            # Reset index to ensure unique indices
            formatted_stats = formatted_stats.reset_index()
            st.dataframe(formatted_stats.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
            
            # Target distribution
            st.subheader("Target Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            df[TARGET_COLUMN].value_counts().plot(kind='bar', ax=ax, color='#2c3e50')
            plt.title('Distribution of Obesity Levels', fontsize=14, pad=20)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Correlation matrix
            st.subheader("Feature Correlation Matrix")
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
            plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
            st.pyplot(fig)
            
        elif page == "Model Performance":
            st.header("Model Performance Analysis")
            
            # Add Custom Random Forest to the model list if it exists in session state
            available_models = list(models.keys())
            if 'custom_rf_model' in st.session_state and st.session_state.custom_rf_model is not None:
                available_models.append("Custom Random Forest")
            
            selected_model = st.selectbox("Select Model", available_models)
            
            if selected_model == "Custom Random Forest":
                # Display Custom Random Forest metrics
                if 'custom_rf_metrics' in st.session_state and st.session_state.custom_rf_metrics is not None:
                    metrics = st.session_state.custom_rf_metrics
                    display_model_metrics(metrics, label_encoders)
                    
                    st.subheader("Confusion Matrix")
                    custom_pred = st.session_state.custom_rf_model.predict(st.session_state.custom_rf_training_data['X_test'])
                    cm = confusion_matrix(st.session_state.custom_rf_training_data['y_test'], custom_pred)
                    fig = plot_confusion_matrix(cm, label_encoders[TARGET_COLUMN].classes_)
                    st.pyplot(fig)
                    
                    # Feature importance plot
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(st.session_state.custom_rf_model, 
                                               st.session_state.custom_rf_training_data['feature_names'])
                    if fig:
                        st.pyplot(fig)
                else:
                    st.warning("Please train the Custom Random Forest model first in the 'Custom Random Forest' tab.")
            else:
                # Display metrics for other models
                metrics_path = os.path.join(MODELS_DIR, f"{selected_model.lower().replace(' ', '_')}_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    display_model_metrics(metrics, label_encoders)
                    
                    st.subheader("Confusion Matrix")
                    cm = np.array(metrics['confusion_matrix'])
                    fig = plot_confusion_matrix(cm, label_encoders[TARGET_COLUMN].classes_)
                    st.pyplot(fig)
                    
                    if metrics.get('roc_curve') is not None:
                        st.subheader("ROC Curve")
                        fpr = metrics['roc_curve']['fpr']
                        tpr = metrics['roc_curve']['tpr']
                        auc = metrics['roc_curve']['auc']
                        fig = plot_roc_curve(fpr, tpr, auc)
                        st.pyplot(fig)
            
        elif page == "Feature Analysis":
            st.header("Feature Analysis")
            
            st.subheader("Feature Importance")
            rf_model = models['Random Forest']
            fig = plot_feature_importance(rf_model, df_processed.columns)
            if fig:
                st.pyplot(fig)
            
            st.subheader("Feature Distributions")
            selected_feature = st.selectbox("Select Feature", df.columns)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=selected_feature, hue=TARGET_COLUMN, multiple="stack", palette='viridis')
            plt.title(f'Distribution of {selected_feature} by Obesity Level', fontsize=14, pad=20)
            st.pyplot(fig)
            
            st.subheader("Feature Relationships")
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X Feature", df.columns)
            with col2:
                y_feature = st.selectbox("Select Y Feature", df.columns)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=TARGET_COLUMN, palette='viridis')
            plt.title(f'Relationship between {x_feature} and {y_feature}', fontsize=14, pad=20)
            st.pyplot(fig)
        
        elif page == "Model Comparison":
            st.header("Model Comparison Analysis")
            
            # Load metrics for all models
            metrics_dict = {}
            for model_name in models.keys():
                metrics_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics_dict[model_name] = json.load(f)
            
            # Overall metrics comparison
            st.subheader("Overall Performance Comparison")
            
            # Create tabs for different comparison views
            comparison_tab1, comparison_tab2, comparison_tab3 = st.tabs([
                "Metrics Comparison", "Confusion Matrices", "Per-Class Performance"
            ])
            
            with comparison_tab1:
                # Plot comparison of main metrics
                metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
                for metric in metrics_to_compare:
                    fig = plot_model_comparison(metrics_dict, metric)
                    st.pyplot(fig)
            
            with comparison_tab2:
                # Plot confusion matrices side by side
                fig = plot_confusion_matrix_comparison(metrics_dict, label_encoders)
                st.pyplot(fig)
            
            with comparison_tab3:
                # Per-class performance comparison
                st.subheader("Per-Class Performance Comparison")
                
                # Create a DataFrame for per-class metrics
                class_metrics_df = pd.DataFrame()
                for model_name, metrics in metrics_dict.items():
                    for metric in ['precision_per_class', 'recall_per_class', 'f1_per_class']:
                        for idx, value in enumerate(metrics[metric]):
                            class_metrics_df = pd.concat([class_metrics_df, pd.DataFrame({
                                'Model': [model_name],
                                'Class': [label_encoders[TARGET_COLUMN].classes_[idx]],
                                'Metric': [metric.replace('_per_class', '')],
                                'Value': [value]
                            })])
                
                # Plot per-class metrics
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=class_metrics_df, x='Class', y='Value', hue='Model', ax=ax)
                plt.xticks(rotation=45)
                plt.title('Per-Class Performance Comparison', fontsize=14, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display detailed comparison table
                st.subheader("Detailed Metrics Table")
                comparison_table = pd.DataFrame()
                for model_name, metrics in metrics_dict.items():
                    model_metrics = pd.DataFrame({
                        'Model': model_name,
                        'Class': label_encoders[TARGET_COLUMN].classes_,
                        'Precision': metrics['precision_per_class'],
                        'Recall': metrics['recall_per_class'],
                        'F1-Score': metrics['f1_per_class']
                    })
                    comparison_table = pd.concat([comparison_table, model_metrics])
                
                # Reset index to ensure unique indices
                comparison_table = comparison_table.reset_index(drop=True)
                
                st.dataframe(comparison_table.style.background_gradient(cmap='YlOrRd', 
                                                                      subset=['Precision', 'Recall', 'F1-Score']),
                           use_container_width=True)
        
        elif page == "Make Prediction":
            st.header("Obesity Level Prediction")
            
            # Create input form and get selected model
            input_data, selected_model = create_input_form()
            
            # Add confidence threshold slider
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.3,
                max_value=0.9,
                value=0.6,
                step=0.1,
                help="Minimum probability threshold for high confidence predictions"
            )
            
            # Add prediction button
            if st.button("Predict Obesity Level"):
                try:
                    if selected_model == "Custom Random Forest":
                        # Check if model exists in session state
                        if 'custom_rf_model' not in st.session_state or st.session_state.custom_rf_model is None:
                            st.error("Please train the Custom Random Forest model first in the 'Custom Random Forest' tab.")
                            return
                        
                        # Calculate BMI
                        height_m = float(input_data['Height'])
                        weight_kg = float(input_data['Weight'])
                        bmi = weight_kg / (height_m ** 2)
                        
                        # Get prediction using custom model
                        input_df = pd.DataFrame([input_data])
                        
                        # Add BMI to input data
                        input_df['BMI'] = bmi
                        
                        # Ensure all required features are present and in correct order
                        required_features = st.session_state.custom_rf_training_data['feature_names']
                        for feature in required_features:
                            if feature not in input_df.columns:
                                input_df[feature] = 0  # Add missing features with default value
                        
                        # Reorder columns to match training data
                        input_df = input_df[required_features]
                        
                        # Preprocess the input data
                        input_processed = preprocess_data(input_df, label_encoders, scaler, is_training=False)
                        
                        # Convert to numpy array and ensure correct shape
                        input_processed = np.array(input_processed)
                        if len(input_processed.shape) == 1:
                            input_processed = input_processed.reshape(1, -1)
                        
                        # Get prediction
                        pred = st.session_state.custom_rf_model.predict(input_processed)[0]
                        pred_proba = st.session_state.custom_rf_model.predict_proba(input_processed)[0]
                        
                        # Convert prediction to original label
                        original_label = label_encoders['NObeyesdad'].inverse_transform([pred])[0]
                        
                        # Create probability dictionary
                        probabilities = dict(zip(label_encoders['NObeyesdad'].classes_, pred_proba))
                        
                        # Display results with confidence assessment and BMI
                        st.subheader("Prediction Results")
                        st.markdown(f"### Custom Random Forest Prediction")
                        
                        # Get the highest probability
                        max_prob = max(probabilities.values())
                        prediction_confidence = f"{max_prob:.1%}"
                        
                        # Determine confidence level
                        if max_prob >= confidence_threshold:
                            confidence_level = "High"
                            confidence_color = "green"
                        elif max_prob >= 0.4:
                            confidence_level = "Medium"
                            confidence_color = "orange"
                        else:
                            confidence_level = "Low"
                            confidence_color = "red"
                        
                        # Display prediction with confidence and BMI
                        st.markdown(f"""
                        **Predicted Obesity Level:** {original_label}
                        
                        **BMI Information:**
                        - Calculated BMI: {bmi:.1f}
                        - BMI Category: {get_bmi_category(bmi)}
                        
                        **Confidence Level:** <span style='color:{confidence_color}'>{confidence_level}</span>
                        **Prediction Confidence:** {prediction_confidence}
                        """, unsafe_allow_html=True)
                        
                        # Display probability distribution
                        display_prediction(original_label, probabilities, "Custom Random Forest", label_encoders, confidence_threshold)
                        
                    else:
                        # Get prediction from selected library model
                        prediction, probabilities = get_prediction(input_data, models[selected_model], label_encoders, scaler, confidence_threshold)
                        display_prediction(prediction, probabilities, selected_model, label_encoders, confidence_threshold)
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.error("Please check your input values and try again.")
        
        elif page == "Custom Random Forest":
            st.header("Custom Random Forest Implementation")
            
            try:
                # Initialize session state variables
                if 'custom_rf_model' not in st.session_state:
                    st.session_state.custom_rf_model = None
                if 'custom_rf_training_data' not in st.session_state:
                    st.session_state.custom_rf_training_data = None
                if 'custom_rf_metrics' not in st.session_state:
                    st.session_state.custom_rf_metrics = None
                if 'library_rf_metrics' not in st.session_state:
                    st.session_state.library_rf_metrics = None
                if 'custom_predictions' not in st.session_state:
                    st.session_state.custom_predictions = None
                if 'library_predictions' not in st.session_state:
                    st.session_state.library_predictions = None
                    
                # Ensure we're using the correct target column
                if 'NObeyesdad' not in df_processed.columns:
                    st.error("Target column 'NObeyesdad' not found in the dataset")
                    return
                    
                # Calculate BMI for all samples
                df_with_bmi = df_processed.copy()
                df_with_bmi['BMI'] = df_with_bmi['Weight'] / (df_with_bmi['Height'] ** 2)
                
                # Add BMI to numerical features
                numerical_features = NUMERICAL_FEATURES + ['BMI']
                
                # Split data
                X = df_with_bmi.drop('NObeyesdad', axis=1)
                y = df_with_bmi['NObeyesdad']
                
                # Store feature names before converting to numpy array
                feature_names = X.columns.tolist()
                
                # Convert to numpy arrays
                X = X.values
                y = y.values
                
                # Use a smaller test size for faster training
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
                
                # Add training button
                st.subheader("Model Training")
                if st.button("Train Custom Random Forest Model"):
                    with st.spinner("Training custom Random Forest model..."):
                        try:
                            # Train custom model
                            custom_rf = train_custom_random_forest(X_train, y_train)
                            
                            if custom_rf is None:
                                st.error("Failed to train the custom Random Forest model")
                                return
                            
                            # Get library model for comparison
                            if 'Random Forest' not in models:
                                st.error("Library Random Forest model not found. Please ensure the model is trained and saved.")
                                return
                                
                            library_rf = models['Random Forest']
                            
                            # Store model and results in session state
                            st.session_state.custom_rf_model = custom_rf
                            st.session_state.custom_rf_training_data = {
                                'X_train': X_train,
                                'y_train': y_train,
                                'X_test': X_test,
                                'y_test': y_test,
                                'feature_names': feature_names
                            }
                            
                            # Calculate predictions for both models
                            custom_pred = custom_rf.predict(X_test)
                            library_pred = library_rf.predict(X_test)
                            
                            # Store predictions in session state
                            st.session_state.custom_predictions = custom_pred
                            st.session_state.library_predictions = library_pred
                            
                            # Calculate and store metrics
                            st.session_state.custom_rf_metrics = {
                                'accuracy': accuracy_score(y_test, custom_pred),
                                'precision': precision_score(y_test, custom_pred, average='weighted'),
                                'recall': recall_score(y_test, custom_pred, average='weighted'),
                                'f1': f1_score(y_test, custom_pred, average='weighted'),
                                'precision_per_class': precision_score(y_test, custom_pred, average=None),
                                'recall_per_class': recall_score(y_test, custom_pred, average=None),
                                'f1_per_class': f1_score(y_test, custom_pred, average=None),
                                'support_per_class': np.bincount(y_test),
                                'classification_report': classification_report(y_test, custom_pred, target_names=label_encoders[TARGET_COLUMN].classes_)
                            }
                            
                            st.session_state.library_rf_metrics = {
                                'accuracy': accuracy_score(y_test, library_pred),
                                'precision': precision_score(y_test, library_pred, average='weighted'),
                                'recall': recall_score(y_test, library_pred, average='weighted'),
                                'f1': f1_score(y_test, library_pred, average='weighted'),
                                'precision_per_class': precision_score(y_test, library_pred, average=None),
                                'recall_per_class': recall_score(y_test, library_pred, average=None),
                                'f1_per_class': f1_score(y_test, library_pred, average=None),
                                'support_per_class': np.bincount(y_test),
                                'classification_report': classification_report(y_test, library_pred, target_names=label_encoders[TARGET_COLUMN].classes_)
                            }
                            
                            st.success("Custom Random Forest model trained successfully!")
                        except Exception as e:
                            st.error(f"Error during model training: {str(e)}")
                            st.error("Please check the data preprocessing steps.")
                            return
                
                # Display results if available
                if 'custom_rf_model' in st.session_state and st.session_state.custom_rf_model is not None:
                    # Model comparison section
                    display_model_comparison(st.session_state.custom_rf_metrics, st.session_state.library_rf_metrics, st.session_state.custom_rf_training_data['feature_names'], st.session_state.custom_rf_model, models['Random Forest'], st.session_state.custom_rf_training_data['X_test'], st.session_state.custom_rf_training_data['y_test'], label_encoders)
                else:
                    st.info("Click the 'Train Custom Random Forest Model' button to train the model and view results.")
            
            except Exception as e:
                st.error(f"An error occurred in the Custom Random Forest tab: {str(e)}")
                st.error("Please make sure the data is properly preprocessed and all required features are present.")
        
        elif page == "RF Comparison":
            st.header("Random Forest Models Comparison")
            
            # Check if Custom Random Forest is trained
            if 'custom_rf_model' not in st.session_state or st.session_state.custom_rf_model is None:
                st.warning("Please train the Custom Random Forest model first in the 'Custom Random Forest' tab.")
            else:
                # Load Random Forest metrics
                rf_metrics_path = os.path.join(MODELS_DIR, 'random_forest_metrics.json')
                if not os.path.exists(rf_metrics_path):
                    st.error("Random Forest metrics file not found. Please ensure the model is properly trained and saved.")
                else:
                    with open(rf_metrics_path, 'r') as f:
                        rf_metrics = json.load(f)
                    
                    # Create tabs for different comparison views
                    comparison_tab1, comparison_tab2, comparison_tab3, comparison_tab4 = st.tabs([
                        "Performance Metrics", "Per-Class Performance", "Feature Importance", "Confusion Matrices"
                    ])
                    
                    with comparison_tab1:
                        st.subheader("Overall Performance Metrics")
                        
                        # Create comparison DataFrame
                        metrics_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
                            'Value': [
                                st.session_state.custom_rf_metrics['accuracy'],
                                st.session_state.custom_rf_metrics['precision'],
                                st.session_state.custom_rf_metrics['recall'],
                                st.session_state.custom_rf_metrics['f1'],
                                float(rf_metrics['accuracy']),
                                float(rf_metrics['precision']),
                                float(rf_metrics['recall']),
                                float(rf_metrics['f1'])
                            ],
                            'Model': ['Custom Random Forest'] * 4 + ['Random Forest'] * 4
                        })
                        
                        # Plot comparison
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=metrics_df, x='Metric', y='Value', hue='Model', ax=ax)
                        plt.title('Performance Metrics Comparison')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display detailed metrics table
                        st.subheader("Detailed Metrics Table")
                        pivot_df = metrics_df.pivot(index='Metric', columns='Model', values='Value')
                        pivot_df['Difference'] = pivot_df['Custom Random Forest'] - pivot_df['Random Forest']
                        st.dataframe(pivot_df.style.background_gradient(
                            cmap='RdYlGn',
                            subset=['Difference'],
                            vmin=-0.1,
                            vmax=0.1
                        ))
                    
                    with comparison_tab2:
                        st.subheader("Per-Class Performance")
                        
                        # Create per-class metrics DataFrame
                        class_metrics_df = pd.DataFrame({
                            'Class': label_encoders[TARGET_COLUMN].classes_,
                            'Custom RF Precision': st.session_state.custom_rf_metrics['precision_per_class'],
                            'Custom RF Recall': st.session_state.custom_rf_metrics['recall_per_class'],
                            'Custom RF F1-Score': st.session_state.custom_rf_metrics['f1_per_class'],
                            'RF Precision': rf_metrics['precision_per_class'],
                            'RF Recall': rf_metrics['recall_per_class'],
                            'RF F1-Score': rf_metrics['f1_per_class']
                        })
                        
                        # Plot per-class metrics
                        metrics = ['Precision', 'Recall', 'F1-Score']
                        for metric in metrics:
                            st.subheader(f"{metric} Comparison")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            x = np.arange(len(class_metrics_df['Class']))
                            width = 0.35
                            
                            ax.bar(x - width/2, class_metrics_df[f'Custom RF {metric}'], width, 
                                  label='Custom Random Forest', color='#2c3e50')
                            ax.bar(x + width/2, class_metrics_df[f'RF {metric}'], width, 
                                  label='Random Forest', color='#3498db')
                            
                            ax.set_ylabel(metric)
                            ax.set_title(f'{metric} by Class')
                            ax.set_xticks(x)
                            ax.set_xticklabels(class_metrics_df['Class'], rotation=45, ha='right')
                            ax.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Display detailed comparison table
                        st.subheader("Detailed Per-Class Metrics")
                        st.dataframe(class_metrics_df.style.background_gradient(
                            cmap='YlOrRd',
                            subset=['Custom RF Precision', 'Custom RF Recall', 'Custom RF F1-Score',
                                   'RF Precision', 'RF Recall', 'RF F1-Score']
                        ))
                    
                    with comparison_tab3:
                        st.subheader("Feature Importance Comparison")
                        
                        # Get feature importance for both models
                        custom_importance = pd.DataFrame({
                            'Feature': st.session_state.custom_rf_training_data['feature_names'],
                            'Importance': st.session_state.custom_rf_model.feature_importances_,
                            'Model': 'Custom Random Forest'
                        })
                        
                        library_importance = pd.DataFrame({
                            'Feature': st.session_state.custom_rf_training_data['feature_names'],
                            'Importance': models['Random Forest'].feature_importances_,
                            'Model': 'Random Forest'
                        })
                        
                        # Combine and sort by average importance
                        combined_importance = pd.concat([custom_importance, library_importance])
                        top_features = combined_importance.groupby('Feature')['Importance'].mean().nlargest(10).index
                        
                        # Plot comparison
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(
                            data=combined_importance[combined_importance['Feature'].isin(top_features)],
                            x='Importance',
                            y='Feature',
                            hue='Model',
                            ax=ax
                        )
                        plt.title('Top 10 Feature Importance Comparison')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                        # Display detailed feature importance table
                        st.subheader("Detailed Feature Importance")
                        pivot_importance = combined_importance.pivot(
                            index='Feature', 
                            columns='Model', 
                            values='Importance'
                        ).sort_values('Random Forest', ascending=False)
                        
                        st.dataframe(pivot_importance.style.background_gradient(
                            cmap='YlOrRd',
                            subset=['Custom Random Forest', 'Random Forest']
                        ))
                    
                    with comparison_tab4:
                        st.subheader("Confusion Matrix Comparison")
                        
                        # Calculate confusion matrices
                        custom_pred = st.session_state.custom_rf_model.predict(st.session_state.custom_rf_training_data['X_test'])
                        library_pred = models['Random Forest'].predict(st.session_state.custom_rf_training_data['X_test'])
                        y_test = st.session_state.custom_rf_training_data['y_test']
                        
                        custom_cm = confusion_matrix(y_test, custom_pred)
                        library_cm = confusion_matrix(y_test, library_pred)
                        
                        # Create figure with two subplots side by side
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                        
                        # Plot Custom Random Forest confusion matrix
                        sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues',
                                  xticklabels=label_encoders[TARGET_COLUMN].classes_,
                                  yticklabels=label_encoders[TARGET_COLUMN].classes_,
                                  ax=ax1)
                        ax1.set_title('Custom Random Forest')
                        ax1.set_xlabel('Predicted Label')
                        ax1.set_ylabel('True Label')
                        
                        # Plot Random Forest confusion matrix
                        sns.heatmap(library_cm, annot=True, fmt='d', cmap='Blues',
                                  xticklabels=label_encoders[TARGET_COLUMN].classes_,
                                  yticklabels=label_encoders[TARGET_COLUMN].classes_,
                                  ax=ax2)
                        ax2.set_title('Random Forest')
                        ax2.set_xlabel('Predicted Label')
                        ax2.set_ylabel('True Label')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add confusion matrix analysis
                        st.markdown("""
                        ### Confusion Matrix Analysis
                        
                        The confusion matrices show:
                        1. **Diagonal Elements**: Number of correct predictions for each class
                        2. **Off-Diagonal Elements**: Number of misclassifications
                        
                        Key observations:
                        - Higher numbers on the diagonal indicate better performance
                        - The Random Forest model typically shows:
                            - More concentrated values on the diagonal
                            - Fewer misclassifications
                            - Better class separation
                        - The Custom Random Forest may show:
                            - More spread-out predictions
                            - More frequent misclassifications
                            - Less clear class boundaries
                        """)
                    
                    # Add overall performance analysis
                    st.markdown("""
                    ### Performance Analysis
                    The Random Forest model shows higher performance metrics because:
                    1. It uses optimized hyperparameters and implementation
                    2. It has better handling of feature interactions
                    3. It employs more sophisticated tree building algorithms
                    
                    The Custom Random Forest implementation, while functional, is a simplified version that:
                    1. Uses basic tree building algorithms
                    2. Has simpler feature selection methods
                    3. May not handle edge cases as effectively
                    """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have run the main.py script first to generate the models and metrics.")

def get_bmi_category(bmi):
    """Get BMI category based on BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obesity Class I"
    elif bmi < 40:
        return "Obesity Class II"
    else:
        return "Obesity Class III"

if __name__ == "__main__":
    main() 