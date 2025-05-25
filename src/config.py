"""
Configuration parameters for the obesity prediction project
"""

# Data parameters
DATA_FILE = 'ObesityDataSet_raw_and_data_sinthetic.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'NObeyesdad'  # Target column name

# Model parameters
MODEL_PARAMS = {
    'Logistic Regression': {
        'C': 1.0,
        'max_iter': 1000,
        'multi_class': 'multinomial'
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'class_weight': 'balanced'
    },
    'SVM': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'probability': True
    }
}

# Feature lists
CATEGORICAL_FEATURES = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

NUMERICAL_FEATURES = [
    'Age',
    'Height',
    'Weight',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE'
]

# Directory names
MODELS_DIR = 'models'
VISUALIZATIONS_DIR = 'visualizations'

# Visualization parameters
PLOT_PARAMS = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn'
}

# Cross-validation parameters
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': RANDOM_STATE
}

# Feature engineering parameters
FEATURE_ENGINEERING = {
    'create_bmi': True,  # Create BMI feature from height and weight
    'create_age_groups': True,  # Create age groups
    'create_activity_score': True  # Create activity score from FAF and TUE
} 