import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./dataset/E-commerce Customer Behavior - Sheet1.csv', sep=',')

# Model mekanistik sederhana
def model(P, t, k):
    return k * (df['Total Spend'].mean() - P)

P0 = 400
t = np.linspace(0, 10, 100)
P = odeint(model, P0, t, args=(0.8,))

plt.plot(t, P)
plt.title('Simulasi Pengeluaran Menuju Nilai Rata-rata')
plt.xlabel('Waktu Simulasi')
plt.ylabel('Total Spend')
plt.show()
