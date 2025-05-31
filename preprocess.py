import pandas as pd
import pathlib

raw_file = 'data/raw/iris.csv'
processed_dir = pathlib.Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading raw data from {raw_file}...")
df = pd.read_csv(raw_file)

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'
df_processed = df[features + [target]]

processed_file = processed_dir / 'iris_processed.csv'
df_processed.to_csv(processed_file, index=False)
print(f"Processed dataset saved to {processed_file}.")