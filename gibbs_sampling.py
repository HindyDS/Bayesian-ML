import pandas as pd
import numpy as np
from joblib import Parallel, delayed


class GibbsSampler:
    def __init__(self, df):
        self.df = df

    def _not_in_use(self):
        pass

    def _sample(self, n_samples):
        self._not_in_use()

        samples = []

        # Randomly draw an initial sample.
        initial = self.df.sample()

        # Randomly draw samples while one of the variable fixed, repeat multiple times.
        for _ in range(n_samples):
            for c in initial.T.sample(frac=1).index:
                v = initial.sample()
                sample = self.df[self.df[c] == v.iloc[0]].sample()
                samples.append(sample)
                initial = sample

        samples = pd.concat(samples)
        return samples.values

    def sample_from_data(self, n_sim, n_samples=1000, n_jobs=5):
        results = Parallel(n_jobs=n_jobs)(delayed(self._sample)(n_samples) for _ in range(n_sim))
        results = np.vstack(results)
        results = pd.DataFrame(results)
        return results
