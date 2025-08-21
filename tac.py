import argparse

from tacomin.build_vocab import main as build_vocab_main
from tacomin.compare import main as compare_main
from tacomin.search import main as search_main
from tacomin.compress import main as compress_main
from tacomin import compress
from tacomin import refine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tacomin')
    parser.add_argument(
        'command', type=str, help='Command to run',
        choices=['vocab', 'compare', 'search', 'compress', 'parquet', 'tall', 'refine']
    )
    parser.add_argument('--input-tar', type=str, help='Input file', default='~/Downloads/fuse-binaries-dec2014.tar.gz')
    parser.add_argument('--output-data', type=str, help='Output data file', default='experiments/results/data.npz')
    parser.add_argument('--output-meta', type=str, help='Output data file', default='experiments/results/meta.pkl')
    parser.add_argument('--counts-only', type=bool, help='Do not build the vocab', default='False')
    parser.add_argument('--output-top', type=str, help='Top file', default='experiments/results/top{k}.npz')
    parser.add_argument('--compare-k', type=int, help='Top k', default=20)
    parser.add_argument('--output-tripples', type=str, help='Tripples file', default='experiments/results/ftripples.parquet')
    parser.add_argument('--top-tar', type=str, help='Top tar file', default='experiments/results/filtered2.tar.gz')
    parser.add_argument('--parquet-dir', type=str, help='Parquet directory', default='experiments/results/parquet2')
    parser.add_argument('--parquet-tall-dir', type=str, help='Parquet directory', default='experiments/results/parquet_tall')
    parser.add_argument('--n-contains', type=int, help='Number of contains', default=5)
    parser.add_argument('--n-move-size', type=int, help='Number of move size', default=5)

    args = parser.parse_args()
    if args.command == 'vocab':
        build_vocab_main(input_file=args.input_tar, output_data_file=args.output_data, output_meta_file=args.output_meta, counts_only=bool(args.counts_only))
    elif args.command == 'compare':
        compare_main(infile=args.output_data, outfile=args.output_top, limit=None, top_k=args.compare_k)
    elif args.command == 'search':
        search_main(x_file=args.output_data, top_file=args.output_top, output_tripples=args.output_tripples)
    elif args.command == 'compress':
        compress_main(input_tarfile=args.input_tar, meta_file=args.output_meta, tripples=args.output_tripples, output_tarfile=args.top_tar)
    elif args.command == 'parquet':
        compress.to_parquet(input_tarfile=args.top_tar, metadata_file=args.output_meta, output_dir=args.parquet_dir)
    elif args.command == 'tall':
        compress.to_tall(input_parquet_dir=args.parquet_dir, output_parquet_dir=args.parquet_tall_dir)
    elif args.command == 'refine':
        refine.main(x_file=args.output_data, top_file=args.output_top, parquet_tall_dir=args.parquet_tall_dir, n_contains=args.n_contains, n_move_size=args.n_move_size, top_k=args.compare_k)




