import pandas as pd
import numpy as np
import math

from tqdm import tqdm
from sklearn.metrics import accuracy_score

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers



class MHSampler:
    def __init__(self, lattice_size = (20, 20), temperature = 293.15, J = 1):
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

df = pd.DataFrame(columns = [str(x) for x in range(20*20)] + ["temperature"])

sampler = MHSampler(lattice_size = (20, 20))

t_max = float(input("Maximum T:"))
t_min = float(input("Minimum T:")) #2.26918531421 - (t_max - 2.26918531421)
n_lat = int(input("Number of lattices:"))

for t in [t_max, t_min]:
    temperature = t
    for _ in range(int(n_lat/5)):
        sampler = MHSampler(lattice_size = (20, 20))
        for _ in range(5):
            lattice = sampler.generate_sample(temperature, iterations =20*20*10)
            df.loc[len(df)] = list(lattice) + [temperature]
        print(temperature, end = '\r')

df.to_csv("data.csv")

def execute(t_max):
    
    #Retrieve the data that was generated
    df = pd.read_csv("data.csv", index_col = 0)
    X = np.array(df[[str(x) for x in range(len(df.columns) - 1)]])
    y = np.array(df['temperature'] > ((t_max+t_min)/2), dtype = int)
    
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
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))
    
    #Evaluate the model on the test data
    y_pred = model.predict(X_test)
    y_pred = [item for sublist in y_pred for item in sublist]
    print(y_pred)
    #yr_pred = [t_min if yp < 2.26918531421 else t_max for yp in y_pred]
    for i in range(len(y_pred)):
        if y_pred[i]<0.5:
            y_pred[i]= math.floor(y_pred[i])
        else:
            y_pred[i] = math.ceil(y_pred[i])
            
    difference = abs(y_pred-y_test)
    
    count = np.sum(difference)  # Count the number of incorrect predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100, '%')
    print('Real temperature:', y_test)
    print('Predicted temperature:', y_pred)
    print('Average real temperature: ', sum(y_test)/len(y_test))
    print('Average predicted temperature: ', sum(y_pred)/len(y_pred))
    print('Difference: ', difference)
    count = 0
    for i in range(len(difference)):
        if difference[i] != 0:
            count += 1
    print(f"Accuracy: {(len(difference) - count)/(len(difference)) * 100} %")
    df_nn = pd.DataFrame(columns = [t_max] + [t_min] + [n_lat] + [accuracy*100])
    df_nn.to_csv('data_nn.csv', mode='a', index=False)
    
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

execute(t_max)
