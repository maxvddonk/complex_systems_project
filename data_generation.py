import pandas as pd
import numpy as np

from tqdm import tqdm

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

temperature = 5

while temperature > .1:
    for _ in range(25):
        lattice = sampler.generate_sample(temperature, iterations = 15 * 15 * 10)
        df.loc[len(df)] = list(lattice) + [temperature]
    temperature -= .1
    print(temperature, end = '\r')

df.to_csv("data.csv")
