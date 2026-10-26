import argparse
import sys
import torch

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
    p.add_argument('--batch_size', '-bs', type=int, default=None,
                   help='Maximum sequences per batch (default: CUDA auto 512; fixed mode 32)')
    p.add_argument('--num_workers', '-nw', type=int, default=2,
                   help='Tokenization workers in fixed mode; CUDA adaptive mode tokenizes each batch in the reader')
    p.add_argument('--memory-preset', choices=['auto', 'off', '2', '4', '8', '16', '24', '32', 'gt32'],
                   default='auto', help='CUDA memory preset in GiB; gt32 uses 64 GiB; off uses fixed batches')
    p.add_argument('--sort-window', type=int, default=4096, help='Maximum records per CUDA sorting window')
    p.add_argument('--max-padded-residues', type=int, default=None,
                   help='Optional lower ceiling on batch size times longest sequence length (CUDA adaptive mode)')
    p.add_argument('--oom-fallback', choices=['error', 'cpu'], default='error',
                   help='Action when a single sequence cannot fit the CUDA batch budget')
    p.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Inference device; auto prefers CUDA, then MPS, then CPU',
    )
    p.add_argument(
        '--detail',
        action='store_true',
        help='Use detailed output format (include probabilities, thresholds, and boundaries where applicable)',
    )
    p.add_argument('--topk', '-tk', type=int, default=1, help='Number of top predictions to emit per sequence')
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
            if args.memory_preset not in ('auto', 'off') or args.oom_fallback != 'error' or args.max_padded_residues is not None:
                raise ValueError('GPU memory presets and OOM fallback apply to standard inference, not --multi')
            if not args.profiles_dir:
                raise ValueError('--profiles_dir must be provided when --multi is enabled')
            stats = inference_precision(
                input_path=args.input_path,
                output_path=args.output_path,
                model=args.model,
                date=args.date,
                profiles_dir=args.profiles_dir,
                detail=args.detail,
                device=args.device,
                topk=args.topk,
            )

        else:
            stats = inference(
                input_path=args.input_path,
                output_path=args.output_path,
                model=args.model,
                date=args.date,
                batch_size=(args.batch_size if args.batch_size is not None else
                            (None if args.device in ('auto', 'cuda') and torch.cuda.is_available()
                             and args.memory_preset != 'off' else 32)),
                num_workers=args.num_workers,
                detail=args.detail,
                device=args.device,
                topk=args.topk,
                memory_preset=args.memory_preset,
                sort_window=args.sort_window,
                max_padded_residues=args.max_padded_residues,
                oom_fallback=args.oom_fallback,
            )
        print(f"Processed {stats['total']} sequences, annotated {stats['annotated']}.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
