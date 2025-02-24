import io
from scipy.sparse import save_npz
from pickle import dump
import tarfile
import pandas as pd
from scipy.sparse import csr_matrix
import time
from tqdm import tqdm

from xlrd.compdoc import CompDocError

# Suppress warning for xlrd
import warnings
# Filter all warning that start with "WARNING"
warnings.filterwarnings('ignore')


def load_from_tar(tar):
    vocab = {}
    words_member = []
    words = []
    t0 = time.time()
    total_time_open = 0
    icounter = 0
    for _, member in enumerate(tar):
        t0_loop = time.time()
        if icounter > 1_000_000:
            break
        if not member.isfile():
            continue
        try:
            buffer = tar.extractfile(member)
            dfs = pd.read_excel(io.BytesIO(buffer.read()), sheet_name=None, header=None, index_col=None, nrows=100)
            t1 = time.time()
            total_time_open += t1 - t0_loop
        except CompDocError:
            continue
        except AttributeError:
            continue
        except IndexError:
            continue
        except Exception as e:
            continue
        for dfi, df in enumerate(dfs.values()):
            # Iterate over all cells
            vals = {str(v)[:20] for v in df.values[:100, :100].flatten()}
            # Update vocab
            vocab_size = len(vocab)
            # Add new words to vocab
            vocab.update({v: vocab_size + i for i, v in
                          enumerate(val[:20] if isinstance(val, str) else val for val in vals if val not in vocab)})
            # Update words
            words.append({vocab[v[:20] if isinstance(v, str) else v] for v in vals})
            words_member.append((member, dfi))
            t2 = time.time()
            total_time = (t2 - t0)
            percentage_open = total_time_open / total_time
            icounter += 1
            items_per_second = icounter / (t2 - t0)
            if icounter % 100 == 0:
                print(f'{icounter}', f'{items_per_second:.1f}', f'{percentage_open:.2%}', member.name,
                      '  ', len(dfs), list(dfs.values())[0].shape, len(vals), len(vocab))
    return vocab, words_member, words


def pack(words_member, words):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for name, sheet in tqdm(zip(words_member, words)):
        indices.extend(list(sheet))
        data.extend([1] * len(sheet))
        indptr.append(len(indices))
    x = csr_matrix((data, indices, indptr), dtype=int)
    return x


def save(x, vocab, words_member, output_data_file, output_meta_file):
    print('Saving data')
    save_npz(output_data_file, x)
    # Save the vocab
    print('Saving meta')
    with open(output_meta_file, 'wb') as f:
        dump((vocab, [(w.name.split('/')[-1], snr) for w, snr in words_member]), f)
    print('Done')


def main(
        input_file='/Users/dirkocoetsee/Downloads/fuse-binaries-dec2014.tar.gz',
        output_data_file='experiments/results/data.npz',
        output_meta_file='experiments/results/meta.pkl'
):
    t0 = time.time()
    # pass over all files and build summary
    file = input_file
    tar_stream = tarfile.open(file, mode='r|*')
    vocab, words_member, words =  load_from_tar(tar_stream)
    print('vocab size', len(vocab))
    print('words size', len(words))
    print('words_member size', len(words_member))
    t1 = time.time()
    print('Time taken', t1 - t0)

    print('Packing')
    x = pack(words_member, words)
    print('x shape', x.shape)
    t2 = time.time()
    print('Time taken', t2 - t1)

    # Save to disk
    save(x, vocab, words_member, output_data_file, output_meta_file)
    t3 = time.time()
    print('Time taken', t3 - t2)
    print('Total time taken', t3 - t0)


if __name__ == '__main__':
    main()