import pandas as pd
from gibbs_sampling import GibbsSampler


if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    df.drop('Id', axis=1, inplace=True)

    # Simulate 1000 times
    n_sim = 10
    gs = GibbsSampler(df)
    results = gs.sample_from_data(n_sim)
