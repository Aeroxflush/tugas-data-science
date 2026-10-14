import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/E-commerce Customer Behavior - Sheet1.csv', sep=',')

# Analisis pengeluaran berdasarkan membership
hasil = df.groupby('Membership Type').agg(
    Jumlah_Pelanggan=('Customer ID', 'count'),
    Rata_Rata_Spend=('Total Spend', 'mean'),
    Rata_Rata_Items=('Items Purchased', 'mean'),
    Rata_Rata_Rating=('Average Rating', 'mean')
).round(2)

print("Analisis berdasarkan Membership:")
print(hasil)

# Visual rata-rata pengeluaran
hasil['Rata_Rata_Spend'].sort_values().plot(kind='barh')

plt.title('Rata-rata Pengeluaran berdasarkan Membership')
plt.xlabel('Rata-rata Total Spend')
plt.ylabel('Membership Type')
plt.tight_layout()
plt.show()

# Rekomendasi berdasarkan hasil
tertinggi = hasil['Rata_Rata_Spend'].idxmax()

print("\nRekomendasi awal:")
print(f"Membership dengan rata-rata pengeluaran tertinggi: {tertinggi}")
