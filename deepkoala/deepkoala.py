import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dataset/DataLoader
class ProteinDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.entries = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as file:
            content = file.read()[1:]
            entries = content.split('\n>')
            return [entry for entry in entries if entry.strip()]

    def tokenize(self, sequence: str):
        aa_vocab = {'<pad>': 0, '<unk>': 1, 'A': 2, 'C': 3, 'D': 4, 'E': 5, 
                    'F': 6, 'G': 7, 'H': 8, 'I': 9, 'K': 10, 'L': 11, 
                    'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 
                    'T': 18, 'V': 19, 'W': 20, 'Y': 21}
        return [aa_vocab.get(aa, 1) for aa in sequence]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        header, sequence = entry.split('\n', 1)
        sequence_name = header.split(' ')[0]
        sequence = self.tokenize(sequence.replace('\n', ''))
        sequence = torch.tensor(sequence, dtype=torch.long)  
        sequence_length = len(sequence) - 1
        
        return sequence_name, sequence, sequence_length


def collate_fn(batch):
    sequence_name, sequences, lengths = zip(*batch)
    padded_query_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequence_name, padded_query_sequences, torch.tensor(lengths, dtype=torch.long)


def get_dataloader(dataset_path: str, batch_size: int, num_workers=0):
    dataset = ProteinDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=num_workers)


# Model
class GRUClassifier(nn.Module):
    def __init__(self, hidden:int, layers:int, n_cls:int):
        super().__init__()
        self.embed = nn.Embedding(22,16,padding_idx=0)
        self.rnn   = nn.GRU(16, hidden, num_layers=layers)
        self.ffn   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.cls   = nn.Linear(hidden, n_cls)
    def forward(self,x,lens):
        x = self.embed(x).permute(1,0,2).contiguous()
        x,_ = self.rnn(x)
        idx = torch.arange(lens.size(0), device=lens.device)
        x  = x[lens, idx]
        return self.cls(self.ffn(x))


# Utilities
def load_ko_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    ko2idx = {ko: info["index"] for ko, info in classes.items()}
    idx2ko = {info["index"]: ko for ko, info in classes.items()}
    thresholds = {ko: info["threshold"] for ko, info in classes.items()}

    return ko2idx, idx2ko, thresholds


def find_newest_date(date_str, folder_path):
    yyyymm_pattern = re.compile(r'^\d{6}$')

    if date_str == "new":
        candidates = [
            name for name in os.listdir(folder_path)
            if yyyymm_pattern.match(name) and os.path.isdir(os.path.join(folder_path, name))
        ]
        if not candidates:
            print("Error: No valid YYYYMM directories found in the folder.")
            sys.exit(1)
        latest = max(candidates)
        print(f"Database version: {latest}")
        return latest

    elif yyyymm_pattern.match(date_str):
        target_path = os.path.join(folder_path, date_str)
        if os.path.isdir(target_path):
            print(f"Database version: {date_str}")
            return date_str
        else:
            print(f"Error: Specified date directory '{date_str}' not found in '{folder_path}'")
            sys.exit(1)
    else:
        print("Error: Invalid date format. Use 'YYYYMM' or 'new'.")
        sys.exit(1)


# Inference
def inference(input_path, output_path, mode, date, batch_size, num_workers, output_format):

    model_resources_path = './resources'
    database_date = find_newest_date(date, model_resources_path)

    ko_config_path = os.path.join(model_resources_path, f'{database_date}/ko_config_{mode}.json')
    weights_path = os.path.join(model_resources_path, f'{database_date}/weights_{mode}.pt')

    ko2idx, idx2ko, threshold = load_ko_config(ko_config_path)

    model = GRUClassifier(128, 2, len(ko2idx)).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    names= []
    predict_labels = []
    probabilities = []
    thresholds = []
    annotate = []
    
    total_count = 0
    annotate_count = 0
    
    model.eval()
    with torch.no_grad():
        dataloader = get_dataloader(dataset_path=input_path, batch_size=batch_size, num_workers=num_workers)
        for sequence_name, sequence, sequence_length in tqdm(dataloader, desc="Inference Progress"):
            sequence = sequence.to(device)
            sequence_length = sequence_length.to(device)
            output = model(sequence, sequence_length)
                
            probability = F.softmax(output, dim=1)
            max_probs, predict_label_idx = torch.max(probability, dim=1)

            max_probs = max_probs.cpu().numpy().tolist()
            predict_label_idx = predict_label_idx.cpu().numpy().tolist()

            for i, j, k in zip(sequence_name, predict_label_idx, max_probs):
                names.append(i)
                probabilities.append(k)
                predict_label = idx2ko[j]
                total_count += 1
                
                if k >= threshold[predict_label]:
                    predict_labels.append(predict_label)
                    thresholds.append(threshold[predict_label])
                    annotate.append('*')
                    annotate_count += 1
                else:
                    predict_labels.append(predict_label)
                    thresholds.append(threshold[predict_label])
                    annotate.append('')

    results = pd.DataFrame(
        {
            'name': names,
            'predict_label': predict_labels,
            'probability': probabilities,
            'threshold': thresholds,
            'annotate': annotate
        }
    )

    if output_format == 'simple':
        results.loc[results['annotate'] != '*', 'predict_label'] = pd.NA
        results = results[['name', 'predict_label']]
        results.to_csv(output_path, index=False)
    else:
        results.to_csv(output_path, index=False)

    print(f'Already processed {total_count} sequences, and annotated {annotate_count} sequences.')


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Please enter input/output path and other parameters.")
    parser.add_argument('--input_path', '-i', required=True, type=str, help='Input file path (fasta format)')
    parser.add_argument('--output_path', '-o', required=True, type=str, help='Output file path')
    parser.add_argument('--mode', '-m', default='full_length', choices=['full_length', 'metagenome'], type=str, help='Complete protein sequence or protein sequence in metagenome')
    parser.add_argument('--date', '-d', default='new', type=str, help='Specify the database version by month in the format YYYYMM')
    parser.add_argument('--batch_size', '-bs', default=64, type=int, help='Batch size')
    parser.add_argument('--num_workers', '-nw', default=0, type=int, help='DataLoader workers for inference')
    parser.add_argument('--output_format', '-of', default='simple', choices=['simple', 'detail'], type=str, help='Output detail level: simple (default) or detail')
    args = parser.parse_args()
    inference(args.input_path, args.output_path, args.mode, args.date, args.batch_size, args.num_workers, args.output_format)
