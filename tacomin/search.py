import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import load_npz


def main(x_file='experiments/results/data.npz', top_file='experiments/results/top20.npz',
         output_tripples='experiments/results/ftripples.parquet'):
    k = 20
    # Load from disk
    print('Loading')
    x = load_npz(x_file)
    topo = np.load(top_file)
    top = topo['arr_0']
    print('Loaded', x.shape, top.shape)

    x_lil = x.tolil()
    x_set = [
        set(row)
        for row in x_lil.rows
    ]

    topa = top
    scores = []

    for i, xs in enumerate(tqdm(x_set)):
        for j in range(k):
            for jj in range(k):
                if jj >= j:
                    continue
                j_ind = topa[i, j]
                jj_ind = topa[i, jj]
                sa1 = x_set[i] & x_set[j_ind]
                sa2 = x_set[i] & x_set[jj_ind]
                inter = (sa1 & sa2)
                intern = len(inter)
                sa1i = len(sa1 - inter)
                sa2i = len(sa2 - inter)
                denom1 = len(x_set[j_ind])
                denom2 = len(x_set[jj_ind])
                denom = len(x_set[i])
                scores.append((i, j_ind, jj_ind, sa1i, sa2i, intern, denom, denom1, denom2))

    # Sorting
    ftripples = [(i, j, k,
                  np.sqrt((sa1i / (denom + 100)) * (sa2i / (denom + 100))),  # Geometric mean of areas
                  #np.sqrt((sa1i / (1)) * (sa2i / (1))),  # Geometric mean of nums
                  #np.sqrt((sa1i / (1)) * (sa2i / (1))) / (denom + 10),  # Geometric mean of nums
                  #(sa1i + sa2i + intern) / (denom + 10),                 # Percentage area
                  sa1i, sa2i, intern, denom, denom1, denom2) for i ,j, k, sa1i, sa2i, intern, denom, denom1, denom2 in scores]

    ftripples = sorted(ftripples, key=lambda x: x[3], reverse=True)
    for i, j, k, score, *rest in ftripples[:20]:
        print(i, j, k, score, rest)

    df = pd.DataFrame(ftripples, columns=['i', 'j', 'k', 'score', 'sa1i', 'sa2i', 'inter', 'denom', 'denom1', 'denom2'])
    df.to_parquet(output_tripples)


if __name__ == '__main__':
    main()