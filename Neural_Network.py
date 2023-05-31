import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

#Retrieve the data that was generated
df = pd.read_csv("data.csv", index_col = 0)
X = np.array(df[[str(x) for x in range(len(df.columns) - 1)]])
y = np.array(df['temperature'])

Size = int(len(X[0]))

# Split the data into train and test sets
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_val, y_val, test_size=0.25, random_state=42)

#Neural network with the hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='linear')
])

#Compile the model
model.compile(optimizer='adam', loss='mse')

#Train the model with the train and validation data
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val))

#Evaluate the model on the test data
y_pred = model.predict(X_test)

y_pred = [item for sublist in y_pred for item in sublist]
difference = abs(y_pred-y_test)
print('Input vector:', X_test[0])
print('Real temperature:', y_test)
print('Predicted temperature:', y_pred)
print('Difference: ', difference)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


        
        
       
