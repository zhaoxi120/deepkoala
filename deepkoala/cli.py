import argparse, sys, torch
from .infer import inference

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_path','-i', required=True)
    p.add_argument('--output_path','-o', required=True)
    p.add_argument('--mode','-m', default='full_length', choices=['full_length','metagenome'])
    p.add_argument('--date','-d', default='latest')
    p.add_argument('--batch_size','-bs', type=int, default=32)
    p.add_argument('--num_workers','-nw', type=int, default=0)
    p.add_argument('--output_format','-of', default='simple', choices=['simple','detail'])
    args = p.parse_args()
    try:
        stats = inference(**vars(args))
        print(f"Processed {stats['total']} sequences, annotated {stats['annotated']}.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
