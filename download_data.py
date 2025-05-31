import os
import pathlib

pathlib.Path('data/raw').mkdir(parents=True, exist_ok=True)

iris_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
raw_path = 'data/raw/iris.csv'

print(f"Downloading Iris dataset to {raw_path}...")
os.system(f"wget -q -O {raw_path} {iris_url}")
print("Download complete.")