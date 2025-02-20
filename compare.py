import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm


def csr_topk_rows(csr_matrix, k):
    n_rows = csr_matrix.shape[0]
    topk_indices = np.full((n_rows, k), -1, dtype=np.int32)

    for i in range(n_rows):
        start, end = csr_matrix.indptr[i], csr_matrix.indptr[i + 1]
        row_data = csr_matrix.data[start:end]
        row_indices = csr_matrix.indices[start:end]
        if len(row_data) > k:
            topk_indices[i, :k] = row_indices[np.argpartition(row_data, k)[:k]]
        else:
            topk_indices[i, :len(row_data)] = row_indices[np.argsort(row_data)]

    return topk_indices


def main(infile='data.npz', outfile='top20.npz', limit=None):
    # Load from disk
    x = load_npz(infile)

    # Select columns where sum is larger than 1
    cols = np.where(x.sum(axis=0) > 1)[1]
    xd = x[:, cols]

    # Get boolean tfidf
    print('convert')
    xdc = xd.tocsc()
    print('sum', xdc.shape)
    N = xdc.shape[1]
    idf = np.log(N / xdc.sum(axis=0))
    print('divide', idf.shape)
    xd = xd.multiply(idf)
    print('done', xd.shape)

    # Get the top 20 columns for each row
    k = 20
    xdl = xd.tolil()
    xdc = xd.T.tocsc()

    chunksize = 1000
    topks = []
    for i in tqdm(range(xdl.shape[0])[::chunksize]):
        if limit and i >= limit:
            break
        dots = np.dot(xdl[i:i+chunksize, :], xdc)
        args = csr_topk_rows(dots, k)
        topks.append(args)

    top = np.concat(topks)
    np.savez_compressed(outfile, top=top)


if __name__ == '__main__':
    main()