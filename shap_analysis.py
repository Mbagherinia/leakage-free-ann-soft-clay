import matplotlib.pyplot as plt
import numpy as np
import os

print("Son Revizyon: Model 4 Etiketi 'Random Split' Olarak Güncellendi...")

save_dir = r"c:\Users\majid\Desktop\YSA"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==============================================================================
# VERİLER
# ==============================================================================
# Model 4 etiketi tamamen objektif ve teknik bir terim olan "Random Split" ile değiştirildi
labels = ['Model 1:\nRandom Forest', 
          'Model 2:\nBase ANN', 
          'Model 3:\nUltimate ANN\n(Proposed Model)', 
          'Model 4:\nData Leakage ANN\n(Random Split)']

rmse_values = [205.82, 135.24, 131.82, 109.86]
mae_values = [142.12, 93.19, 87.95, 69.70]

r2_math_values = [0.6063, 0.8300, 0.8385, 0.9217] 
r2_text_labels = ['0.6063', '0.83', '0.8385', '0.9217']

x = np.arange(len(labels))
width = 0.3

# ==============================================================================
# GRAFİK ÇİZİMİ
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(14, 8), dpi=300)

# BARLAR
rects1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE (kPa)', color='#3498db', edgecolor='black', zorder=2)
rects2 = ax1.bar(x + width/2, mae_values, width, label='MAE (kPa)', color='#27ae60', edgecolor='black', zorder=2)

# MODEL 4 İÇİN ÇİZGİLİ TARAMA (HATCHING)
rects1[3].set_hatch('//')
rects2[3].set_hatch('//')

ax1.set_ylabel('Prediction Error: RMSE & MAE (kPa)', fontsize=14, fontweight='bold', color='black')
ax1.set_ylim(0, 280) 
ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')

# KIRMIZI ÇİZGİ
ax2 = ax1.twinx()
ax2.plot(x, r2_math_values, color='#e74c3c', marker='o', markersize=12, linewidth=4, label='R-Squared ($R^2$)', zorder=4)
ax2.set_ylim(0.4, 1.05) 

# Sağ ekseni gizle
ax2.get_yaxis().set_visible(False)
ax2.spines['right'].set_visible(False)

# ==============================================================================
# RAKAMLARI YAZDIRMA
# ==============================================================================
def autolabel_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333')

for i, (val, txt) in enumerate(zip(r2_math_values, r2_text_labels)):
    ax2.annotate(txt,
                xy=(x[i], val),
                xytext=(0, 12),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='#c0392b')

autolabel_bars(rects1)
autolabel_bars(rects2)

# ==============================================================================
# ESTETİK, BAŞLIK VE LEJANT
# ==============================================================================
plt.title("Performance Benchmarking of Predictive Models", fontsize=18, fontweight='bold', y=1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# LEJANT TAM BAŞLIĞIN ALTINDA (Uzun Dikdörtgen Formu)
ax1.legend(lines1 + lines2, labels1 + labels2, 
           loc='upper center',          
           bbox_to_anchor=(0.5, 0.98), 
           ncol=3,                     # Yan yana 3 öğe
           fontsize=12, 
           framealpha=1,               
           edgecolor='black',          
           borderpad=0.8)              

plt.tight_layout()

# Kaydetme
file_path = os.path.join(save_dir, "Figure_8_Benchmark_Performance_Perfect_Math_Final_v5.png")
plt.savefig(file_path, bbox_inches='tight')
plt.show()

print(f"--- BAŞARILI: Model 4 etiketi güncellendi: {file_path} ---")
