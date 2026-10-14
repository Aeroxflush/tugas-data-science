
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/E-commerce Customer Behavior - Sheet1.csv', sep=',')

# Distribusi Total Spend
sns.histplot(data=df, x='Total Spend', kde=True)
plt.title('Distribusi Total Spend')
plt.show()

# Hubungan Total Spend dengan Items Purchased
sns.scatterplot(data=df, x='Items Purchased', y='Total Spend',
                hue='Membership Type')
plt.title('Hubungan Items Purchased dan Total Spend')
plt.show()
