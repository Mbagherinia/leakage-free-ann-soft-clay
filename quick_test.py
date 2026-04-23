import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import time
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("="*70)
print("--- QUICK EXECUTION TEST (FOR TECHNICAL REVIEW ONLY) ---")
print("NOTE: This script trains a miniature, fast-running network on a")
print("tiny subset (200 rows) of the data simply to verify that the coding")
print("environment and pipeline execute without errors.")
print("The predictions here WILL NOT match the high-accuracy results")
print("reported in the paper. To replicate the paper's exact results,")
print("please run the full model in 'four_models.py'.")
print("="*70)

start_time = time.time()

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "ANN_Ready_Data.xlsx")

print(f"\nLoading first 200 rows from:\n{data_path}...")
df = pd.read_excel(data_path, nrows=200)

features = ['NaOH_%', 'KOH_%', 'Curing_Time_Day', 'Water_Content_%', 'Bulk_Density_g_cm3', 'Strain_%', 'Strain_Squared', 'Strain_Cubed']
X = df[features]
y = df['Stress_kPa']

print("Applying StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training the miniature validation framework...")
model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(X_scaled, y)

print("\n--- Quick Prediction Results (Dummy Outputs) ---")
predictions = model.predict(X_scaled[:5])

for i in range(5):
    print(f"Sample {i+1} | Actual Stress: {y.iloc[i]:.2f} kPa | Dummy Prediction: {predictions[i]:.2f} kPa")

print(f"\n--- Quick Test Completed Successfully without errors in {time.time() - start_time:.2f} seconds! ---")