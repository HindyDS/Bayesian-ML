import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class MetropolisHastings:
    def __init__(self, df):
        self.df = df

    def _not_in_use(self):
        pass

    def _sample(self, n_samples=1000):
        self._not_in_use()

        samples = self.df.sample(n_samples, replace=True)
        return samples.values

    def sample_from_data(self, n_sim, n_samples=1000, n_jobs=5):
        results = Parallel(n_jobs=n_jobs)(delayed(self._sample)(n_samples) for _ in range(n_sim))
        results = np.vstack(results)
        results = pd.DataFrame(results)
        return results
