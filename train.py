import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import pathlib

data_file = 'data/processed/iris_processed.csv'
models_dir = pathlib.Path('data/models')
models_dir.mkdir(parents=True, exist_ok=True)
model_file = models_dir / 'iris_rf_model.pkl'

print(f"Loading processed data from {data_file}...")
df = pd.read_csv(data_file)
X = df.drop('species', axis=1)
y = df['species']

print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}.")