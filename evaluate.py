import pandas as pd
import pickle
from sklearn.metrics import classification_report

data_file = 'data/processed/iris_processed.csv'
model_file = 'data/models/iris_rf_model.pkl'

print(f"Loading test data from {data_file}...")
df = pd.read_csv(data_file)
X = df.drop('species', axis=1)
y = df['species']

print(f"Loading model from {model_file}...")
with open(model_file, 'rb') as f:
    model = pickle.load(f)

print("Evaluating model...")
preds = model.predict(X)
report = classification_report(y, preds)
print(report)