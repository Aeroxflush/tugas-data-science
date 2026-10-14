
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('./dataset/E-commerce Customer Behavior - Sheet1.csv', sep=',')

# Stats deskriptif
print("Statistik Deskriptif:")
print(df[['Age', 'Total Spend', 'Items Purchased',
          'Average Rating', 'Days Since Last Purchase']].describe())

print("\nDistribusi Satisfaction Level:")
print(df['Satisfaction Level'].value_counts())

# Visual distribusi
sns.countplot(data=df, x='Satisfaction Level')
plt.title('Distribusi Tingkat Kepuasan Pelanggan')
plt.xlabel('Satisfaction Level')
plt.ylabel('Jumlah Pelanggan')
plt.show()

# Persiapan data
fitur = ['Age', 'Total Spend', 'Items Purchased',
         'Average Rating', 'Days Since Last Purchase']

data = df[fitur + ['Satisfaction Level']].dropna()

X = data[fitur]
encoder = LabelEncoder()
y = encoder.fit_transform(data['Satisfaction Level'])

# (Training dan testing) Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Melatih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
prediksi = model.predict(X_test)

print("\nAkurasi Prediksi:",
      round(accuracy_score(y_test, prediksi), 3))

print("\nClassification Report:")
print(classification_report(
    y_test, prediksi,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

# Visual tingkat kepentingan fitur
importance = pd.Series(
    model.feature_importances_, index=fitur
).sort_values()

importance.plot(kind='barh')
plt.title('Tingkat Kepentingan Fitur dalam Prediksi')
plt.xlabel('Nilai Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()
