import pandas as pd
import tarfile
from tqdm import tqdm
from pickle import load


def main(input_tarfile, meta_file, tripples, output_tarfile):
    # Iterate through the tar file and build another tar file with only the
    # tripples mentioned in the output_tripples file

    # Load the tripples
    tripples = pd.read_parquet(tripples)

    # Load the meta file
    with open(meta_file, 'rb') as f:
        vocab, words_member = load(f)

    # Create a set of used indices
    used_indices = {i for i in tripples['i'].values} | {i for i in tripples['j'].values} | {i for i in tripples['k'].values}
    print('Used indices', len(used_indices))

    # Create a set of tar names to save
    tar_names = {filename for i, (filename, sheetnr) in enumerate(words_member) if i in used_indices}
    print('Tar names', len(tar_names))

    # Open the tar file in stream mode, and open the output tar file also in stream mode
    with tarfile.open(input_tarfile, mode='r|*') as tar_stream, tarfile.open(output_tarfile, mode='w|gz') as output_tar_stream:
        for tar_info in tqdm(tar_stream):
            if tar_info.name.split('/')[-1] in tar_names:
                output_tar_stream.addfile(tar_info, tar_stream.extractfile(tar_info))

    print('Done')
