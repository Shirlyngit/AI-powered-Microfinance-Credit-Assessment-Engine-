import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, classification_report, confusion_matrix,
    roc_curve, auc
)

# Set style for plots
sns.set_style("whitegrid")

# --- 1. Load Data ---
# Since you pasted a sample, we'll simulate loading it from a string.
# In your actual environment, you'd use:
df = pd.read_csv('sampled.csv')

# Define a Credit Risk column based on CreditScore (example threshold)
# You might need to adjust this threshold based on domain expertise or business requirements.
# For simplicity, let's assume a CreditScore below 600 indicates 'High Risk' (1) and above is 'Low Risk' (0).
df['CreditRisk'] = (df['CreditScore'] < 600).astype(int)

# Drop the original CreditScore column as it's directly used to create the target
df = df.drop('CreditScore', axis=1)

print("Initial DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("\nDataFrame Description:")
print(df.describe())

# --- 2. Assigning Meaningful Categorical Labels ---
print("\n--- Assigning Meaningful Categorical Labels ---")

education_mapping = {
    0: 'Primary',
    1: 'Secondary',
    2: 'Undergraduate',
    3: 'Postgraduate'
}

employment_type_mapping = {
    0: 'Unemployed',
    1: 'Salaried',
    2: 'Self-Employed',
    3: 'Contract-Part-time'
}

marital_status_mapping = {
    0: 'Single',
    1: 'Married',
    2: 'Divorced/Widowed'
}

loan_purpose_mapping = {
    0: 'Debt Consolidation',
    1: 'Home Improvement',
    2: 'Business',
    3: 'Education',
    4: 'Other-Miscellaneous'
}

df['Education'] = df['Education'].map(education_mapping)
df['EmploymentType'] = df['EmploymentType'].map(employment_type_mapping)
df['MaritalStatus'] = df['MaritalStatus'].map(marital_status_mapping)
df['LoanPurpose'] = df['LoanPurpose'].map(loan_purpose_mapping)

print("\nDataFrame Head after mapping categorical values:")
print(df.head())
print("\nUnique values after mapping:")
print("Education:", df['Education'].unique())
print("EmploymentType:", df['EmploymentType'].unique())
print("MaritalStatus:", df['MaritalStatus'].unique())
print("LoanPurpose:", df['LoanPurpose'].unique())

# --- 3. Data Cleaning ---
print("\n--- Data Cleaning ---")
print("Missing values before cleaning:\n", df.isnull().sum())

# Handle missing values (if any) - for this sample, there are none, but good practice
# For numerical features, common strategies include mean, median, or mode imputation
# For categorical features, mode imputation or a 'Missing' category
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object': # Categorical
            df[col].fillna(df[col].mode()[0], inplace=True)
        else: # Numerical
            df[col].fillna(df[col].median(), inplace=True) # Using median for numerical

print("\nMissing values after cleaning:\n", df.isnull().sum())

# Check for duplicates
print(f"\nNumber of duplicate rows before dropping: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Number of duplicate rows after dropping: {df.duplicated().sum()}")

# --- 4. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis (EDA) ---")

# Target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='CreditRisk', data=df)
plt.title('Distribution of Credit Risk (Target Variable)')
plt.xlabel('Credit Risk (0: Low Risk, 1: High Risk)')
plt.ylabel('Count')
plt.show()
print(f"\nCredit Risk Distribution:\n{df['CreditRisk'].value_counts(normalize=True)}")

# Numerical features distribution
numerical_cols = ['Age', 'Income', 'MonthsEmployed', 'DTIRatio']
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, len(numerical_cols), i + 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Box plots for numerical features vs. Credit Risk
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, len(numerical_cols), i + 1)
    sns.boxplot(x='CreditRisk', y=col, data=df)
    plt.title(f'{col} vs. Credit Risk')
plt.tight_layout()
plt.show()

# Categorical features distribution
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
plt.figure(figsize=(20, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 4, i + 1)
    sns.countplot(x=col, data=df, hue='CreditRisk', palette='viridis')
    plt.title(f'Distribution of {col} by Credit Risk')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Correlation Matrix (for numerical features initially)
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# --- 5. Data Preprocessing (for Machine Learning) ---
print("\n--- Data Preprocessing ---")

# Separate features (X) and target (y)
X = df.drop('CreditRisk', axis=1)
y = df['CreditRisk']

# Identify numerical and categorical features for preprocessing pipeline
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create a preprocessing pipeline
# StandardScaler for numerical features, OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any, like original index if not dropped)
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify for imbalanced target

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 6. Model Training and Evaluation ---
print("\n--- Model Training and Evaluation ---")

models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    # Create a pipeline for each model that includes preprocessing
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class

    # Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{name} Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }

    # For credit scoring, the primary goal is classification (predicting risk category).
    # RMSE and R2 score are typically used for regression tasks (predicting continuous values).
    # If you were predicting the exact CreditScore value, these would be relevant.
    # However, since we defined 'CreditRisk' as a binary target, classification metrics are paramount.
    # We can still look at how well the *probabilities* correlate with the actual outcome,
    # but that's essentially what ROC AUC measures.
    # If you still want RMSE/R2 on probabilities (which is less conventional for binary classification)
    # let's assume we use them to assess the 'calibration' of probabilities.
    # This is a bit of a stretch for pure binary classification evaluation:
    # rmse_prob = np.sqrt(mean_squared_error(y_test, y_prob))
    # r2_prob = r2_score(y_test, y_prob)
    # print(f"RMSE (on probabilities): {rmse_prob:.4f}")
    # print(f"R2 Score (on probabilities): {r2_prob:.4f}")


# --- 7. Cross-Validation to Improve Performance & Robustness ---
print("\n--- Cross-Validation ---")

cv_results = {}
n_splits = 5 # Number of folds for K-Fold cross-validation

for name, model in models.items():
    print(f"\nPerforming {n_splits}-Fold Cross-Validation for {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Evaluate using roc_auc as it's often a good metric for imbalanced classification
    scores = cross_val_score(pipeline, X, y, cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), scoring='roc_auc', n_jobs=-1)
    
    cv_results[name] = {
        'ROC AUC Mean': scores.mean(),
        'ROC AUC Std': scores.std(),
        'All Scores': scores
    }
    print(f"{name} - ROC AUC Cross-Validation Scores: {scores}")
    print(f"{name} - Mean ROC AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- 8. Compare Model Metrics ---
print("\n--- Model Comparison ---")
comparison_df = pd.DataFrame(results).T
print("\nTest Set Performance Comparison:")
print(comparison_df)

print("\nCross-Validation ROC AUC Comparison:")
cv_comparison_df = pd.DataFrame({
    'Model': [name for name in cv_results.keys()],
    'Mean ROC AUC': [res['ROC AUC Mean'] for res in cv_results.values()],
    'ROC AUC Std': [res['ROC AUC Std'] for res in cv_results.values()]
}).set_index('Model')
print(cv_comparison_df)

# Determine the better model based on mean ROC AUC from cross-validation
best_model_name = cv_comparison_df['Mean ROC AUC'].idxmax()
print(f"\nBased on Mean ROC AUC from {n_splits}-Fold Cross-Validation, the better model is: {best_model_name}")

# --- (Optional) Further Steps: Hyperparameter Tuning ---
print("\n--- Further Steps (Optional): Hyperparameter Tuning ---")
print("To further improve model performance, hyperparameter tuning techniques like GridSearchCV or RandomizedSearchCV can be applied.")
print("Example for RandomForestClassifier (conceptual):")
print("""
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
                              param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search_rf.fit(X, y) # Fit on full data for best hyperparameters
print(f"Best parameters for RandomForest: {grid_search_rf.best_params_}")
print(f"Best ROC AUC score for RandomForest: {grid_search_rf.best_score_}")
""")

print("\nExample for XGBClassifier (conceptual):")
print("""
from sklearn.model_selection import GridSearchCV

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.7, 0.9],
    'classifier__colsample_bytree': [0.7, 0.9]
}

grid_search_xgb = GridSearchCV(Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
                             param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search_xgb.fit(X, y)
print(f"Best parameters for XGBoost: {grid_search_xgb.best_params_}")
print(f"Best ROC AUC score for XGBoost: {grid_search_xgb.best_score_}")
""")




# ... (your existing ML training script code) ...

# --- 9. Save the Trained Model Pipeline ---
import joblib

# After training both models, let's decide which one to save.
# Based on your cross-validation results, pick the best one.
# For demonstration, let's assume XGBClassifier is chosen as the final model due to better performance often.

# Retrain the chosen model on the full training data (or best from GridSearchCV if you implemented it)
# For simplicity, we'll use the pipeline that was fit during the loop.
# If you run hyperparameter tuning, you'd save grid_search_xgb.best_estimator_

# Choose the model you want to deploy
final_model_name = best_model_name # Based on your comparison logic, e.g., 'XGBClassifier'
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', models[final_model_name])])

# Re-fit the final pipeline on the entire training set (X_train, y_train) for production readiness,
# or if you prefer, fit on full X, y if you believe it generalizes better after tuning.
# For this example, let's just save the pipeline that was already fit in the loop for the final model.
# If you want to train it on the whole dataset for final deployment, you would do:
# final_pipeline.fit(X, y) # Fit on ALL data (X, y) after selecting hyperparameters

# To ensure the pipeline used for saving is trained on X_train, y_train:
# The `pipeline` variable inside the loop refers to the one currently being processed.
# So, if you want to save the `XGBClassifier` pipeline, you'd ensure it's the one currently in `pipeline`.
# A more robust way is to re-create and fit it, or store the best pipeline object directly.

# Let's re-create and fit the best pipeline explicitly for saving:
if final_model_name == 'XGBClassifier':
    best_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
elif final_model_name == 'RandomForestClassifier':
    best_classifier = RandomForestClassifier(random_state=42)
else:
    raise ValueError("Selected model not found.")

final_pipeline_to_save = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', best_classifier)])

# Fit the chosen final pipeline on the training data
final_pipeline_to_save.fit(X_train, y_train) # Use X_train, y_train or X, y depending on your strategy

MODEL_FILE_NAME = "credit_scoring_pipeline.pkl"
joblib.dump(final_pipeline_to_save, MODEL_FILE_NAME)
print(f"\nSaved trained model pipeline as {MODEL_FILE_NAME}")

# Also save the categorical mappings separately, as they are Python dicts
# and will be needed for preprocessing new input data in FastAPI
categorical_mappings = {
    'education_mapping': education_mapping,
    'employment_type_mapping': employment_type_mapping,
    'marital_status_mapping': marital_status_mapping,
    'loan_purpose_mapping': loan_purpose_mapping
}
joblib.dump(categorical_mappings, "categorical_mappings.pkl")
print("Saved categorical mappings as categorical_mappings.pkl") 