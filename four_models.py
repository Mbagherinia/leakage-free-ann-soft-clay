import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split # Data Leakage (Rastgele Bölme) için gerekli

# 1. LOAD THE PREPARED DATASET
file_path = r"c:\Users\majid\Desktop\YSA\ANN_Ready_Data.xlsx"
df = pd.read_excel(file_path)

# ==============================================================================
# LEAKAGE-FREE SPLITTING (GROUPKFOLD LOGIC) - FOR MODELS 1, 2, and 3
# ==============================================================================
unique_samples = df['Sample_ID'].unique()
np.random.seed(42) # Ensure the exact same 19 samples are selected for fair comparison
test_samples = np.random.choice(unique_samples, size=int(len(unique_samples)*0.2), replace=False)

test_df = df[df['Sample_ID'].isin(test_samples)]
train_df = df[~df['Sample_ID'].isin(test_samples)]

y_train_honest = train_df['Stress_kPa']
y_test_honest = test_df['Stress_kPa']

# DEFINE FEATURE SETS
base_features = ['NaOH_%', 'KOH_%', 'Curing_Time_Day', 'Water_Content_%', 'Bulk_Density_g_cm3', 'Strain_%']
ultimate_features = base_features + ['Strain_Squared', 'Strain_Cubed']

print("Training all 4 models for the ultimate benchmarking... Please wait...\n")

# ==============================================================================
# MODEL 1: RANDOM FOREST REGRESSOR (Tree-Based Approach / Leakage-Free)
# ==============================================================================
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(train_df[ultimate_features], y_train_honest)
y_pred_rf = np.maximum(0, rf_model.predict(test_df[ultimate_features]))

r2_rf = r2_score(y_test_honest, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_honest, y_pred_rf))
mae_rf = mean_absolute_error(y_test_honest, y_pred_rf)

# ==============================================================================
# MODEL 2: BASE ANN (Standard Neural Network / Leakage-Free)
# ==============================================================================
scaler_base = StandardScaler()
X_train_base = scaler_base.fit_transform(train_df[base_features])
X_test_base = scaler_base.transform(test_df[base_features])

ann_base = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', 
                        max_iter=2000, random_state=42, early_stopping=True)
ann_base.fit(X_train_base, y_train_honest)
y_pred_ann_base = np.maximum(0, ann_base.predict(X_test_base))

r2_ann_base = r2_score(y_test_honest, y_pred_ann_base)
rmse_ann_base = np.sqrt(mean_squared_error(y_test_honest, y_pred_ann_base))
mae_ann_base = mean_absolute_error(y_test_honest, y_pred_ann_base)

# ==============================================================================
# MODEL 3: ULTIMATE ANN (Champion Model with Full Feature Engineering / Leakage-Free)
# ==============================================================================
scaler_ult = StandardScaler()
X_train_ult = scaler_ult.fit_transform(train_df[ultimate_features])
X_test_ult = scaler_ult.transform(test_df[ultimate_features])

ann_ult = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', 
                       max_iter=2000, random_state=42, early_stopping=True)
ann_ult.fit(X_train_ult, y_train_honest)
y_pred_ann_ult = np.maximum(0, ann_ult.predict(X_test_ult))

r2_ann_ult = r2_score(y_test_honest, y_pred_ann_ult)
rmse_ann_ult = np.sqrt(mean_squared_error(y_test_honest, y_pred_ann_ult))
mae_ann_ult = mean_absolute_error(y_test_honest, y_pred_ann_ult)

# ==============================================================================
# MODEL 4: DATA LEAKAGE ANN (The Mistake of the Literature - Random Split)
# ==============================================================================
# Here we do NOT use GroupKFold. We randomly split rows like other papers do.
X_all = df[ultimate_features]
y_all = df['Stress_kPa']

X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

scaler_leak = StandardScaler()
X_train_leak_scaled = scaler_leak.fit_transform(X_train_leak)
X_test_leak_scaled = scaler_leak.transform(X_test_leak)

ann_leak = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', 
                        max_iter=2000, random_state=42, early_stopping=True)
ann_leak.fit(X_train_leak_scaled, y_train_leak)
y_pred_leak = np.maximum(0, ann_leak.predict(X_test_leak_scaled))

r2_leak = r2_score(y_test_leak, y_pred_leak)
rmse_leak = np.sqrt(mean_squared_error(y_test_leak, y_pred_leak))
mae_leak = mean_absolute_error(y_test_leak, y_pred_leak)

# ==============================================================================
# PRINT Q1 JOURNAL READY BENCHMARKING TABLE
# ==============================================================================
print("="*105)
print("TABLE: ALGORITHM AND FEATURE ENGINEERING BENCHMARKING RESULTS (Including Data Leakage)")
print("="*105)
print(f"{'Algorithm / Approach':<55} | {'R-Squared (R2) %':<16} | {'RMSE (kPa)':<12} | {'MAE (kPa)':<10}")
print("-" * 105)
print(f"{'1. Random Forest (Tree-Based Model - Leakage Free)':<55} | {r2_rf*100:>14.2f} % | {rmse_rf:>10.2f} | {mae_rf:>8.2f}")
print(f"{'2. Base ANN (No Physical Polynomials - Leakage Free)':<55} | {r2_ann_base*100:>14.2f} % | {rmse_ann_base:>10.2f} | {mae_ann_base:>8.2f}")
print(f"{'3. Ultimate ANN (Champion Model - Leakage Free)':<55} | {r2_ann_ult*100:>14.2f} % | {rmse_ann_ult:>10.2f} | {mae_ann_ult:>8.2f}")
print("-" * 105)
print(f"{'4. DATA LEAKAGE ANN (Literature Mistake - Random Split)':<55} | {r2_leak*100:>14.2f} % | {rmse_leak:>10.2f} | {mae_leak:>8.2f}")
print("="*105)
save_dir = r"c:\Users\majid\Desktop\YSA"

print("\nEğitilmiş modeller .pkl formatında masaüstüne kaydediliyor...")

# Model 1: Random Forest (Scaler kullanmadığımız için doğrudan kaydediyoruz)
joblib.dump(rf_model, os.path.join(save_dir, 'Model_1_RandomForest.pkl'))

# Model 2: Base ANN (Model ve Scaler'ı tek bir sözlük içinde paketliyoruz)
joblib.dump({'model': ann_base, 'scaler': scaler_base}, 
            os.path.join(save_dir, 'Model_2_Base_ANN.pkl'))

# Model 3: Ultimate ANN (ŞAMPİYON MODEL - Makaleye yüklenecek olan)
joblib.dump({'model': ann_ult, 'scaler': scaler_ult}, 
            os.path.join(save_dir, 'Model_3_Ultimate_ANN.pkl'))

# Model 4: Data Leakage ANN (Literatür Hatası Modeli)
joblib.dump({'model': ann_leak, 'scaler': scaler_leak}, 
            os.path.join(save_dir, 'Model_4_DataLeakage_ANN.pkl'))

print("Tebrikler! 4 Model de (gerekli scaler'lar ile birlikte) başarıyla kaydedildi.")
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# ULTIMATE ANN İÇİN FEATURE IMPORTANCE (ÖZELLİK ÖNEM DERECESİ) ANALİZİ
# ==============================================================================
print("\nUltimate ANN modeli için Permütasyon Duyarlılık Analizi hesaplanıyor... Lütfen bekleyin...")

# Şampiyon modelimiz (ann_ult) üzerinde permütasyon analizi yapıyoruz
result = permutation_importance(ann_ult, X_test_ult, y_test_honest, n_repeats=10, 
                                random_state=42, scoring='r2')

# Özelliklerin isimleri ve önem dereceleri (Ortalama R2 düşüşü)
importances = result.importances_mean
importances = np.maximum(importances, 0) 

# Yüzdelik dilime çevirme
importances_percent = 100.0 * (importances / importances.sum())

# Büyükten küçüğe sıralama
sorted_idx = np.argsort(importances_percent)
sorted_features = np.array(ultimate_features)[sorted_idx]
sorted_importances = importances_percent[sorted_idx]

pretty_names = {
    'NaOH_%': 'NaOH (%)',
    'KOH_%': 'KOH (%)',
    'Curing_Time_Day': 'Curing Time (Days)',
    'Water_Content_%': 'Water Content (%)',
    'Bulk_Density_g_cm3': 'Bulk Density (g/cm³)',
    'Strain_%': 'Strain (%)',
    'Strain_Squared': 'Strain² (%)',
    'Strain_Cubed': 'Strain³ (%)'
}
sorted_features_pretty = [pretty_names.get(feat, feat) for feat in sorted_features]

# ==============================================================================
# Q1 FORMATINDA BAR GRAFİĞİ ÇİZİMİ
# ==============================================================================
plt.figure(figsize=(10, 6), dpi=300)
bars = plt.barh(sorted_features_pretty, sorted_importances, color='#4c72b0', edgecolor='black')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', va='center', ha='left', fontsize=11, fontweight='bold')

plt.xlabel('Relative Importance (%)', fontsize=12, fontweight='bold')
plt.title('Permutation Feature Importance for Ultimate ANN Model', fontsize=14, fontweight='bold', pad=15)
plt.xlim(0, max(sorted_importances) + 10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

kayit_yolu = r"c:\Users\majid\Desktop\YSA\Figure_W_Feature_Importance.png"
plt.savefig(kayit_yolu, format='png', bbox_inches='tight')
plt.show()

print(f"--- BAŞARILI: Özellik Önem Derecesi grafiği şuraya kaydedildi: {kayit_yolu} ---")
