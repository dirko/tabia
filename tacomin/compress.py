import os
import glob
import io
import pandas as pd
import pickle
import tarfile
from tqdm import tqdm
from pickle import load
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def main(input_tarfile, meta_file, tripples, output_tarfile):
    # Iterate through the tar file and build another tar file with only the
    # tripples mentioned in the output_tripples file

    # Load the tripples
    tripples = pd.read_parquet(tripples)
    ftripples = tripples[tripples.score > 0]

    # Load the meta file
    with open(meta_file, 'rb') as f:
        vocab, words_member = load(f)

    # Create a set of used indices
    used_indices = {i for i in ftripples['i'].values} | {i for i in ftripples['j'].values} | {i for i in ftripples['k'].values}
    print('Used indices', len(used_indices))

    # Create a set of tar names to save
    tar_names = {filename for i, (filename, sheetnr) in enumerate(words_member) if i in used_indices}
    print('Tar names', len(tar_names))

    # Open the tar file in stream mode, and open the output tar file also in stream mode
    with tarfile.open(input_tarfile, mode='r|*') as tar_stream, tarfile.open(output_tarfile, mode='w|') as output_tar_stream:
        for i, tar_info in enumerate(tqdm(tar_stream)):
            if tar_info.name.split('/')[-1] in tar_names:
                output_tar_stream.addfile(tar_info, tar_stream.extractfile(tar_info))

    print('Done')


def components(df, n_top=1000):
    """Remove similar sheets from the top n_top sheets by only
    keeping the highest scoring sheet per connected component"""

    # Create the three edge sets from each 3-clique
    edges_ij = df[['i', 'j']]
    edges_ik = df[['i', 'k']].rename(columns={'k': 'j'})
    edges_jk = df[['j', 'k']].rename(columns={'j': 'i', 'k': 'j'})

    # Concatenate all edges into one DataFrame
    edges = pd.concat([edges_ij, edges_ik, edges_jk], ignore_index=True)

    # For an undirected graph, ensure symmetry by adding the reversed pairs
    edges_sym = pd.concat([edges, edges.rename(columns={'i': 'j', 'j': 'i'})], ignore_index=True)

    # Extract row and column indices for the sparse matrix
    rows = edges_sym['i']
    cols = edges_sym['j']

    # Determine the total number of nodes. Here we assume nodes are 0-indexed.
    n_nodes = df[['i', 'j', 'k']].max().max() + 1

    # Build the sparse adjacency matrix
    data = [1] * len(rows)
    adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Compute the connected components
    n_components, labels = connected_components(adjacency_matrix, directed=False)

    print("Number of connected components:", n_components)
    print("Component labels for each node:", labels)
    df['component'] = [labels[i].item() for i in df.i]

    # Now get the highest scoring row per component
    dfc = df.head(n_top).groupby('component').apply(lambda x: x.loc[x.score.idxmax()]).sort_values('score', ascending=False)
    return dfc


def to_parquet(input_tarfile, metadata_file, output_dir):
    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_file, 'rb') as f:
        vocab, words_member = pickle.load(f)
    member_map = {member: i for i, member in enumerate(words_member)}

    tar_stream = tarfile.open(input_tarfile, mode='r|*')
    vocab = {}
    icounter = 0
    total_counter = 0
    df_list = []
    for member_number, member in enumerate(tar_stream):
        if icounter > 1_000_001:
            break
        if not member.isfile():
            continue
        try:
            buffer = tar_stream.extractfile(member)
            dfs = pd.read_excel(io.BytesIO(buffer.read()), sheet_name=None, header=None, index_col=None, nrows=100)
        except Exception as e:
            continue
        for dfi, df in enumerate(dfs.values()):
            # Iterate over all cells
            vals = {str(v)[:20] for v in df.values[:100, :100].flatten()}
            # Update vocab
            vocab_size = len(vocab)
            # Add new words to vocab
            # 0 Is empty, 1 is Other (rare)
            vocab.update({v: vocab_size + i + 2 for i, v in
                          enumerate(val[:20] if isinstance(val, str) else val for val in vals if val not in vocab)})
            # Make sure the df is 100 by 100
            df_clipped = df.iloc[:100, :100]
            df_padded = df_clipped
            # ..also by adding empty cols if necessary
            if df_clipped.shape[1] < 100:
                df_padded = pd.concat([df_clipped, pd.DataFrame(columns=range(df_clipped.shape[1], 100))], axis=1)
            # Map df to vocab - all ints
            df_mapped = df_padded.map(lambda x: vocab[str(x)[:20]] if str(x)[:20] in vocab else 1)
            df_mapped['f'] = member.name
            df_mapped['s'] = dfi
            df_mapped['fs'] = member_map[member.name.split('/')[-1], dfi]
            df_mapped['rows'] = df_clipped.shape[0]
            df_mapped['cols'] = df_clipped.shape[1]
            # Add row index as column, excluding the index name
            df_mapped.index.name = 'i'
            df_mapped.reset_index(drop=False, inplace=True)

            # Add to list
            df_list.append(df_mapped)

            icounter += 1

            # Accumulate sheets and write to parquet
            if len(df_list) >= 1000:
                dfc = pd.concat(df_list)
                dfc.columns = dfc.columns.astype(str)
                print(f'{icounter}', f'{total_counter}', '  ', len(dfs), len(vocab), dfc.shape)
                dfc.to_parquet(f'{output_dir}/df_{total_counter:04d}.parquet', compression='zstd')
                df_list = []
                total_counter += 1

    # Save last batch
    dfc = pd.concat(df_list)
    dfc.columns = dfc.columns.astype(str)
    print(f'{icounter}', f'{total_counter}', '  ', len(dfs), len(vocab), dfc.shape)
    dfc.to_parquet(f'{output_dir}/df_{total_counter:04d}.parquet', compression='zstd')

    # Save vocab
    vocab_df = pd.DataFrame(vocab.items(), columns=['word', 'id'])
    vocab_df.to_parquet(f'{output_dir}/vocab.parquet', compression='zstd')

    # Write
    dfm = pd.DataFrame(words_member, columns=['word', 'member'])
    dfm['index'] = dfm.index
    dfm['filename'] = dfm.word.apply(lambda x: f'cc-binaries/{x}')
    dfm.to_parquet(f'{output_dir}/words_member.parquet')


def to_tall(input_parquet_dir, output_parquet_dir):
    # Read the parquet files which have schema (f, fs, rows, cols, col1, col2, ...)
    # And transform to parquet files with schema (fs, i, j, value)
    # Read them one by one, because they cannot all fit into memory
    # Create output dir if not exists
    vocab = pd.read_parquet(os.path.join(input_parquet_dir, 'vocab.parquet'))
    nan_index = vocab[vocab.word == 'nan'].id.values[0]

    os.makedirs(output_parquet_dir, exist_ok=True)
    parquet_files = glob.glob(os.path.join(input_parquet_dir, 'df_*.parquet'))
    for parquet_file in tqdm(parquet_files):
        df = pd.read_parquet(parquet_file)
        cols = [f'{i}' for i in range(100)]
        dfp = df.set_index(['i', 'fs'])[cols]\
            .stack().reset_index()\
            .rename(columns={'level_2': 'j', 0: 'v'})[['fs', 'i', 'j', 'v']]
        dfp = dfp[dfp.v != nan_index]
        dfp = dfp.astype({'i': 'int16', 'fs': 'int32', 'j': 'uint8', 'v': 'int32'})
        dfp.sort_values(['fs', 'i', 'j'], inplace=True)
        dfp.to_parquet(os.path.join(output_parquet_dir, os.path.basename(parquet_file)), compression='zstd')
