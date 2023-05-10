import pandas as pd
import numpy as np

from tqdm import tqdm

from PIL import Image

class MHSampler:
    def __init__(self, lattice_size = (10, 10), temperature = 293.15, J = 1):
        self.lattice = np.ones(lattice_size)
        self.lattice += np.random.randint(2, size = lattice_size) * -2

        self.T = temperature

        self.J = J

    def _energy(self, lattice):
        """
        non-public function that returns a new total energy corresponding to the
        state of some lattice by looping over all possible pairs of neighbours
        """
        counted_pairs = []

        E = 0

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if {(i, j), ((i + 1) % lattice.shape[0], j)} not in counted_pairs:
                    E -= self.J * lattice[(i, j)] * lattice[((i + 1) % lattice.shape[0], j)]
                    counted_pairs.append({(i, j), ((i + 1) % lattice.shape[0], j)})
                if {(i, j), ((i - 1) % lattice.shape[1], j)} not in counted_pairs:
                    E -= self.J * lattice[(i, j)] * lattice[((i - 1) % lattice.shape[0], j)]
                    counted_pairs.append({(i, j), ((i - 1) % lattice.shape[1], j)})
                if {(i, j), (i, (j + 1) % lattice.shape[1])} not in counted_pairs:
                    E -= self.J * lattice[(i, j)] * lattice[(i, (j + 1) % lattice.shape[1])]
                    counted_pairs.append({(i, j), (i, (j + 1) % lattice.shape[1])})
                if {(i, j), (i, (j - 1) % lattice.shape[1])} not in counted_pairs:
                    E -= self.J * lattice[(i, j)] * lattice[(i, (j - 1) % lattice.shape[1])]
                    counted_pairs.append({(i, j), (i, (j - 1) % lattice.shape[1])})

        return E

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
        if iterations == None:
            iterations = len(self.lattice.flatten()) * 10

        E0 = self._energy(self.lattice)
        for i in range(iterations):
            new = self._generate_new_state()
            E1 = self._energy(new)
            if E1 <= E0:
                self.lattice = new
                E0 = E1
            elif np.random.uniform() < np.exp(-(E1 - E0) / self.T):
                self.lattice = new
                E0 = E1

    def generate_sample(self, temperature, iterations = None):
        self.T = temperature

        self._mh(iterations = iterations)

        return self.lattice.flatten()

df = pd.DataFrame(columns = [str(x) for x in range(100)] + ["temperature"])

sampler = MHSampler()

for _ in tqdm(range(500)):
    temperature = np.random.uniform(0.1, 10)

    lattice = sampler.generate_sample(temperature)

    df.loc[len(df)] = list(lattice) + [temperature]

df.to_csv("data.csv")
