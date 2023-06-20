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


Delta_1 =  float(input("Delta 1: "))

Delta_2 =  float(input("Delta 2: "))

Delta_3 =  float(input("Delta 3: "))

Delta_4 =  float(input("Delta 4: "))

def execute(Delta):
    
    t_min = 2.26918531421 - Delta
    t_max = 2.26918531421 + Delta
    print("Minimum:", t_min)
    print("Maximum:", t_max)
    
    for t in [t_max, t_min]:
        temperature = t
        for j in range(5000):
            sampler = MHSampler(lattice_size = (20, 20))
            for i in range(10):
                lattice = sampler.generate_sample(temperature, iterations = 20 * 20 * 10)
                df.loc[len(df)] = list(lattice) + [temperature]
        print(temperature, end = '\r')
    
    df.to_csv("data_train.csv")
    #Retrieve the train data that was generated
    df_train = pd.read_csv("data_train.csv", index_col = 0)
    X = np.array(df_train[[str(x) for x in range(len(df_train.columns) - 1)]])
    y = np.array(df_train['temperature'] > 2.26918531421, dtype = int)
    
    #Retrieve the test data that was generated
    df_test = pd.read_csv("data_test.csv", index_col = 0)
    X_test = np.array(df_test[[str(x) for x in range(len(df_test.columns) - 1)]])
    y_test = np.array(df_test['temperature'] > 2.26918531421, dtype = int)

    # Split the data into train and test sets
    X_val, X_train, y_val, y_train = train_test_split(X, y, test_size=0.2, random_state=42)
        
    #Neural network with the hidden layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    #Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Train the model with the train and validation data
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))
    Amount_of_test_data = 400
    Accuracy = np.zeros(int(len(y_test)/(Amount_of_test_data/2)))
    for j in range(0, len(y_test), Amount_of_test_data):
        #Evaluate the model on the test data
        y_pred = model.predict(X_test[j:j+Amount_of_test_data])
        y_pred = [item for sublist in y_pred for item in sublist]
        y_pred = np.round(y_pred).astype(int)
        
        Accuracy[j//Amount_of_test_data] = accuracy_score(y_test[j:j+Amount_of_test_data], y_pred)*100
        Accuracy[-(j//Amount_of_test_data) - 1] = Accuracy[j//Amount_of_test_data]
    
    x = np.zeros(int(len(y_test)/(Amount_of_test_data/2)))
    for i in range(1,46):
        x[i-1] = i/20
        x[-i] = 2 * 2.26918531421-x[i-1]
        
    print((Accuracy))
    plt.ylim(0, 100)    
    plt.plot(x, Accuracy, label="\u03B4 = {}".format(Delta))

execute(Delta_1)
execute(Delta_2)
execute(Delta_3)
execute(Delta_4)
plt.xlabel("Temperature")
plt.ylabel("Accuracy in %")
plt.legend()
plt.show()