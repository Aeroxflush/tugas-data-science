import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/E-commerce Customer Behavior - Sheet1.csv', sep=',')

# Stats deskriptif
print(df[['Age', 'Total Spend', 'Items Purchased',
          'Average Rating', 'Days Since Last Purchase']].describe())

# Masing2 pengguna membership
print(df['Membership Type'].value_counts())

# Visualisasi
sns.boxplot(x='Membership Type', y='Total Spend', data=df)
plt.title('Distribusi Total Spend berdasarkan Membership Type')
plt.show()
