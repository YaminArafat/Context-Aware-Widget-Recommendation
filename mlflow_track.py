import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

mlflow.set_experiment("Multi_Label_Classification_Monitoring")

df = pd.read_csv("./data/synthetic/synthetic_multilabel_dataset.csv")

feature_cols = [
    'Time of Day', 'Activity', 'Age', 'Heart Rate', 
    'Weather', 'Day of Week', 'Recent Events', 'Battery Status', 'Sleep Score'
]

target_cols = [
    'Daily Activity', 'Sleep', 'Running', 'Swimming', 'Calories Burn', 
    'Heart Rate (show)', 'Steps', 'Consumed', 'Weather (show)', 'Chance of Rain', 
    'Events', 'Battery'
]

X = df[feature_cols]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = ['Time of Day', 'Activity', 'Weather', 'Day of Week', 'Recent Events']
numerical_cols = ['Age', 'Heart Rate', 'Battery Status', 'Sleep Score']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

n_estimators = 15
max_depth = 2
if mlflow.active_run():
    mlflow.end_run()
with mlflow.start_run(run_name="Random_Forest_MultiOutput"):
    
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("classifier_type", "RandomForest")
    
    base_classifier = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MultiOutputClassifier(base_classifier))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    mlflow.log_metric("exact_match_accuracy", accuracy)
    mlflow.log_metric("hamming_loss", h_loss)
    mlflow.log_metric("f1_score_macro", f1_macro)

    mlflow.sklearn.log_model(
        sk_model=model_pipeline, 
        artifact_path="widget_prediction_pipeline"
    )
    
    print(f"Accuracy: {accuracy:.4f} | Hamming Loss: {h_loss:.4f} | F1 Macro: {f1_macro:.4f}")