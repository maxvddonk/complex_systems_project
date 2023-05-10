import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df = pd.read_csv("data.csv", index_col = 0)

states = np.array(df[[str(x) for x in range(100)]])

df['magnetic_moment'] = np.abs(np.sum(states, axis = 1))

print(f"correlation between temperature and magnetic moment (absolute value of" + \
      f"sum of values in lattice):\t{df.corr()['temperature']['magnetic_moment']:.3f}")

plt.scatter(df['temperature'], df['magnetic_moment'], alpha = .5)
plt.xlabel("T")
plt.ylabel("magnetic moment")
plt.show()
