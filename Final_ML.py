# Install required libraries
# pip install catboost pymongo pandas

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB and load the data
client = MongoClient("mongodb+srv://tanakitpitak:hua300548@dataci.cxist.mongodb.net/?retryWrites=true&w=majority&appName=Datasci")
db = client["Datasci"]
collection = db["final"]
df = pd.DataFrame(list(collection.find())).drop(columns=['_id'], errors='ignore')

# Prepare the data
X = df.drop(columns=['Unnamed: 0', 'impact'], errors='ignore')
y = df['impact']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Encode labels and compute class weights
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

# Define and train the CatBoost model with specific parameters
catboost_model = CatBoostClassifier(
    depth=8,
    learning_rate=0.01,
    iterations=1500,
    l2_leaf_reg=9,
    class_weights=class_weights,
    loss_function='MultiClass',
    random_state=42,
    verbose=0
)

catboost_model.fit(X_train, y_train_encoded)

# Make predictions
y_pred_encoded = catboost_model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Print results
print("Classification Report:\n", classification_report(y_test, y_pred))