import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Neural network with the hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
    ])

#Compile the model
model.compile(optimizer='adam', loss='mse')

#Retrieve the data that was generated
df = pd.read_csv("data.csv", index_col = 0)
X = np.array(df[[str(x) for x in range(len(df.columns) - 1)]])
y = np.array(df['temperature'])

#Train the model on the generated data
model.fit(X, y, epochs=10, batch_size=32)

#Evaluate the model on some input data
X_test = np.array([np.ones(100) +  np.random.randint(2, size = 100) * -2])         
y_test = model.predict(X_test)


#Plot the input data into an image
plt.imshow(X_test.reshape(10,10),cmap='gray')
plt.show()

print('Input vector:', X_test)
print('Predicted temperature:', y_test[0][0])
