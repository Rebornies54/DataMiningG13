# Obesity Level Prediction Analysis

This project implements a machine learning system to predict obesity levels based on eating habits and physical conditions. The system uses various classification algorithms and provides an interactive dashboard for analysis.

## Project Structure

```
DataMiningG13/
├── src/
│   ├── app.py              # Streamlit dashboard application
│   ├── config.py           # Configuration settings
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── model_training.py   # Model training and evaluation
│   └── main.py            # Main execution script
├── models/                 # Trained model files
├── visualizations/         # Generated visualizations
└── ObesityDataSet_raw_and_data_sinthetic.csv  # Dataset
```

## Data Processing Workflow

1. **Data Loading and Preprocessing**
   - Load raw dataset from CSV
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
   - Split data into training and testing sets

2. **Feature Engineering**
   - Feature selection based on importance
   - Handling of categorical variables using label encoding
   - Normalization of numerical features

3. **Model Training and Evaluation**
   - Training multiple classification algorithms
   - Cross-validation
   - Hyperparameter tuning
   - Performance metrics calculation
   - Model persistence

## Implemented Algorithms

### 1. Random Forest
- **How it works**: 
  - Creates multiple decision trees using random subsets of features
  - Each tree makes a prediction, and the final prediction is based on majority voting
  - Handles both numerical and categorical features
  - Provides feature importance scores

### 2. Support Vector Machine (SVM)
- **How it works**:
  - Finds the optimal hyperplane to separate classes
  - Uses kernel trick for non-linear classification
  - Effective in high-dimensional spaces
  - Robust against overfitting

### 3. Neural Network
- **How it works**:
  - Multi-layer perceptron architecture
  - Uses backpropagation for training
  - Can learn complex patterns
  - Includes dropout for regularization

## Performance Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve and AUC

## Dashboard Features

1. **Data Overview**
   - Basic statistics
   - Target distribution
   - Feature correlation matrix

2. **Model Performance**
   - Overall metrics
   - Per-class performance
   - Confusion matrix
   - ROC curves

3. **Feature Analysis**
   - Feature importance
   - Feature distributions
   - Feature relationships

## Setup and Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the main script to train models:
```bash
python src/main.py
```

3. Launch the dashboard:
```bash
streamlit run src/app.py
```

## Requirements

- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- joblib



Atribute Information

The attributes related with eating habits are:

Frequent consumption of high caloric food (FAVC)
Frequency of consumption of vegetables(FCVC)
Number of main meals (NCP)
Consumption of food between meals (CAEC)
Consumption of water daily (CH20)
Consumption of alcohol (CALC)

The attributes related with the physical condition are:

Calories consumption monitoring (SCC)
Physical activity frequency (FAF)
Time using technology devices (TUE)
Transportation used (MTRANS)

Other variables obtained were:

Gender
Age
Height
Weight
Family history with overweight
SMOKE

NObeyesdad (Target variable) was created with the values of: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III
