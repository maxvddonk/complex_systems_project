import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df = pd.read_csv("data.csv", index_col = 0)

states = np.array(df[[str(x) for x in range(len(df.columns) - 1)]])

df['magnetic_moment'] = np.abs(np.sum(states, axis = 1)) / (15 * 15)

print(f"correlation between temperature and magnetic moment (absolute value of " + \
      f"sum of values in lattice):\t{df.corr()['temperature']['magnetic_moment']:.3f}")

temps = np.unique(df['temperature'])
avg_moments = np.zeros(len(temps))

for i, temp in enumerate(temps):
    avg_moments[i] = np.mean(df.loc[df['temperature'] == temp]['magnetic_moment'])

plt.plot(temps, avg_moments)
plt.xlabel("T")
plt.ylabel("magnetic moment")
plt.show()
