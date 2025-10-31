import argparse
import sys

from .infer import inference
from .infer_multi import inference_precision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_path', '-i', required=True)
    p.add_argument('--output_path', '-o', required=True)
    p.add_argument('--model', '-m', default='full', choices=['full', 'frag'])
    p.add_argument('--date', '-d', default='latest')
    p.add_argument('--batch_size', '-bs', type=int, default=32)
    p.add_argument('--num_workers', '-nw', type=int, default=2)
    p.add_argument('--output_format', '-of', default='simple', choices=['simple', 'detail'])
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
                output_format=args.output_format,
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
                output_format=args.output_format,
            )
        print(f"Processed {stats['total']} sequences, annotated {stats['annotated']}.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
