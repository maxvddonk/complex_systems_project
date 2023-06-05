import pandas as pd
import numpy as np

from tqdm import tqdm

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers



class MHSampler:
    def __init__(self, lattice_size = (10, 10), temperature = 293.15, J = 1):
        self.lattice = np.ones(lattice_size)
        self.lattice += np.random.randint(2, size = lattice_size) * -2

        self.T = temperature

        self.J = J

        #self.Energy = self._energy(self.lattice)

    """def _energy(self, lattice):
        E = 0

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                #these conditions ensure that no pairs get counted twice
                if (i + 1) % lattice.shape[0] > i:
                    E -= self.J * lattice[(i, j)] * lattice[((i + 1) % lattice.shape[0], j)]
                if (i - 1) % lattice.shape[0] > i:
                    E -= self.J * lattice[(i, j)] * lattice[((i - 1) % lattice.shape[0], j)]
                if (j + 1) % lattice.shape[0] > j:
                    E -= self.J * lattice[(i, j)] * lattice[(i, (j + 1) % lattice.shape[1])]
                if (j - 1) % lattice.shape[0] > j:
                    E -= self.J * lattice[(i, j)] * lattice[(i, (j - 1) % lattice.shape[1])]

        return E"""

    def _delta_energy(self, new_lattice):
        """
        non-public function that returns the difference between current energy
        and the energy of a newly proposed changed lattice
        """
        indices = np.where(self.lattice != new_lattice)
        i = indices[0][0]
        j = indices[1][0]

        old_energy = 0
        old_energy -= self.J * self.lattice[(i, j)] * self.lattice[((i + 1) % self.lattice.shape[0], j)]
        old_energy -= self.J * self.lattice[(i, j)] * self.lattice[((i - 1) % self.lattice.shape[0], j)]
        old_energy -= self.J * self.lattice[(i, j)] * self.lattice[(i, (j + 1) % self.lattice.shape[1])]
        old_energy -= self.J * self.lattice[(i, j)] * self.lattice[(i, (j - 1) % self.lattice.shape[1])]

        new_energy = 0
        new_energy -= self.J * new_lattice[(i, j)] * new_lattice[((i + 1) % new_lattice.shape[0], j)]
        new_energy -= self.J * new_lattice[(i, j)] * new_lattice[((i - 1) % new_lattice.shape[0], j)]
        new_energy -= self.J * new_lattice[(i, j)] * new_lattice[(i, (j + 1) % new_lattice.shape[1])]
        new_energy -= self.J * new_lattice[(i, j)] * new_lattice[(i, (j - 1) % new_lattice.shape[1])]

        return new_energy - old_energy

    def _generate_new_state(self):
        """
        non-public function that returns a new overall state of the current
        lattice by changing the state of a single cell
        """
        shape = self.lattice.shape

        flattened = self.lattice.flatten()

        index = np.random.randint(len(flattened))

        flattened[index] *= -1

        return flattened.reshape(shape)

    def _mh(self, iterations = None):
        """
        function that performs the metropolis hastings algorithm
        """
        if iterations == None:
            iterations = len(self.lattice.flatten()) * 10

        #main loop
        for i in range(iterations):
            new = self._generate_new_state()
            E_delta = self._delta_energy(new)

            if E_delta <= 0:
                self.lattice = new
                #self.Energy = self.Energy + E_delta
            elif np.random.uniform() < np.exp(-(E_delta) / self.T):
                self.lattice = new
                #self.Energy = self.Energy + E_delta

    def generate_sample(self, temperature, iterations = None):
        """
        returns a 'snapshot' of the lattice after performing the Metropolis-Hastings
        algorithm given a certain temperature
        """
        self.T = temperature

        self._mh(iterations = iterations)

        return self.lattice.flatten()

df = pd.DataFrame(columns = [str(x) for x in range(15*15)] + ["temperature"])

sampler = MHSampler(lattice_size = (15, 15))

t_max = float(input("Maximum T:"))
t_min = float(input("Minimum T:"))

for t in [t_max, t_min]:
    temperature = t
    for _ in range(50):
        lattice = sampler.generate_sample(temperature, iterations = 15 * 15 * 10)
        df.loc[len(df)] = list(lattice) + [temperature]
    print(temperature, end = '\r')

df.to_csv("data.csv")

def execute(t_max,t_min):
    #Retrieve the data that was generated
    df = pd.read_csv("data.csv", index_col = 0)
    X = np.array(df[[str(x) for x in range(len(df.columns) - 1)]])
    y = np.array(df['temperature'] > 2.26918531421, dtype = int)
    
    print(y)
    Size = int(len(X[0]))
    
    # Split the data into train and test sets
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_val, y_val, test_size=0.25, random_state=42)
    
    #Neural network with the hidden layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    #Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    #Train the model with the train and validation data
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
    
    #Evaluate the model on the test data
    y_pred = model.predict(X_test)
    y_pred = [item for sublist in y_pred for item in sublist]
    #yr_pred = [t_min if yp < 2.26918531421 else t_max for yp in y_pred]
    
    difference = abs(y_pred-y_test)
    
    count = np.sum(difference)  # Count the number of incorrect predictions
    accuracy = (len(difference) - count) / len(difference) * 100
    print(f"Accuracy: {accuracy}%")
    
    print('Input vector:', X_test[0])
    print('Real temperature:', y_test)
    print('Predicted temperature:', y_pred)
    # print('Difference: ', difference)
    # count = 0
    # for i in range(len(difference)):
    #     if difference[i] != 0:
    #         count += 1
    # print(f"Accuracy: {(len(difference) - count)/(len(difference)) * 100} %")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

execute(t_max,t_min)
        
