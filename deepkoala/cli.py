import argparse
import sys

from .infer import inference
from .infer_multi import inference_precision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_path', '-i', required=True, help='Input FASTA file containing sequences to annotate')
    p.add_argument('--output_path', '-o', required=True, help='Destination path for inference results (CSV)')
    p.add_argument(
        '--model',
        '-m',
        default='full',
        choices=['full', 'frag'],
        help="Model variant to use: 'full' for full-length sequences or 'frag' for fragments",
    )
    p.add_argument('--date', '-d', default='latest', help='Model date tag; use "latest" for the newest release')
    p.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size for batched inference')
    p.add_argument('--num_workers', '-nw', type=int, default=2, help='Number of dataloader worker processes')
    p.add_argument(
        '--detail',
        action='store_true',
        help='Use detailed output format (include probabilities, thresholds, and boundaries where applicable)',
    )
    p.add_argument('--top_k', '-tk', type=int, default=1, help='Number of top predictions to emit per sequence')
    p.add_argument('--multi', action='store_true', help='Enable multi-domain mode for domain annotations')
    p.add_argument(
        '--profiles_dir',
        '-pd',
        default='',
        help='Directory containing KO-specific HMM profiles (multi-domain mode only)',
    )
    args = p.parse_args()
    try:
        if args.multi:
            if not args.profiles_dir:
                raise ValueError('--profiles_dir must be provided when --multi is enabled')
            profiles_dir = args.profiles_dir
            stats = inference_precision(
                input_path=args.input_path,
                output_path=args.output_path,
                model=args.model,
                date=args.date,
                profiles_dir=profiles_dir,
                detail=args.detail,
                top_k=args.top_k,
            )

        else:
            if args.profiles_dir:
                args.profiles_dir = None

            stats = inference(
                input_path=args.input_path,
                output_path=args.output_path,
                model=args.model,
                date=args.date,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                detail=args.detail,
                top_k=args.top_k,
            )
        print(f"Processed {stats['total']} sequences, annotated {stats['annotated']}.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
