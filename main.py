import io
import tarfile
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def main():
    # pass over all files and build summary
    file = '/Users/dirkocoetsee/Downloads/fuse-binaries-dec2014.tar.gz'
    # Get list of files in tar archive
    tar = tarfile.open(file)
    for i, member in enumerate(tar.getmembers()):
        if i > 10:
            break
        print(member.name)
        buffer = tar.extractfile(member)
        # Read the excel file in the buffer
        df = pd.read_excel(io.BytesIO(buffer.read()))
        print('  ', df.shape)





if __name__ == '__main__':
    main()