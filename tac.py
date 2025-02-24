import argparse

from tacomin.build_vocab import main as build_vocab_main
from tacomin.compare import main as compare_main
from tacomin.search import main as search_main
from tacomin.compress import main as compress_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tacomin')
    parser.add_argument('command', type=str, help='Command to run', choices=['vocab', 'compare', 'search', 'compress'])
    parser.add_argument('--input-tar', type=str, help='Input file', default='/Users/dirkocoetsee/Downloads/fuse-binaries-dec2014.tar.gz')
    parser.add_argument('--output-data', type=str, help='Output data file', default='experiments/results/data.npz')
    parser.add_argument('--output-meta', type=str, help='Output data file', default='experiments/results/meta.pkl')
    parser.add_argument('--output-top', type=str, help='Top file', default='top20.npz')
    parser.add_argument('--output-tripples', type=str, help='Tripples file', default='experiments/results/ftripples.parquet')
    parser.add_argument('--top-tar', type=str, help='Top tar file', default='experiments/results/filtered.tar.gz')

    args = parser.parse_args()
    if args.command == 'vocab':
        build_vocab_main(input_file=args.input_tar, output_data_file=args.output_data, output_meta_file=args.output_meta)
    elif args.command == 'compare':
        compare_main(infile=args.output_data, outfile=args.output_top)
    elif args.command == 'search':
        search_main(x_file=args.output_data, top_file=args.output_top, output_tripples=args.output_tripples)
    elif args.command == 'compress':
        compress_main(input_tarfile=args.input_tar, meta_file=args.output_meta, tripples=args.output_tripples, output_tarfile=args.top_tar)



