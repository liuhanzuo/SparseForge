#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
import numpy as np
import tiktoken

SEP = "\n\n"

def format_example(ex, name):
    if name == 'alpaca':
        instr = ex.get('instruction') or ''
        inp = ex.get('input') or ''
        out = ex.get('output') or ''
        if inp.strip():
            text = f"Instruction: {instr}\nInput: {inp}\nResponse: {out}{SEP}"
        else:
            text = f"Instruction: {instr}\nResponse: {out}{SEP}"
        return text
    elif name == 'dolly':
        instr = ex.get('instruction') or ''
        ctx = ex.get('context') or ''
        rsp = ex.get('response') or ''
        if ctx.strip():
            text = f"Instruction: {instr}\nContext: {ctx}\nResponse: {rsp}{SEP}"
        else:
            text = f"Instruction: {instr}\nResponse: {rsp}{SEP}"
        return text
    else:
        raise ValueError('unknown dataset name')


def build_token_array(texts):
    enc = tiktoken.get_encoding('gpt2')
    ids = []
    for t in texts:
        ids.extend(enc.encode_ordinary(t))
    arr = np.array(ids, dtype=np.uint16)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['alpaca','dolly'], required=True)
    ap.add_argument('--out_base', default='data')
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--max_samples', type=int, default=0, help='0 for all')
    args = ap.parse_args()

    if args.dataset == 'alpaca':
        ds = load_dataset('tatsu-lab/alpaca')
        split_name = 'train'
    elif args.dataset == 'dolly':
        ds = load_dataset('databricks/databricks-dolly-15k')
        split_name = 'train'

    data = ds[split_name]
    if args.max_samples and args.max_samples > 0:
        data = data.select(range(min(args.max_samples, len(data))))

    texts = [format_example(ex, args.dataset) for ex in data]

    # split
    n = len(texts)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    train_texts = texts[:n_train]
    val_texts = texts[n_train:]

    # tokenize
    train_ids = build_token_array(train_texts)
    val_ids = build_token_array(val_texts)

    out_dir = os.path.join(args.out_base, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # save memmaps
    train_bin = os.path.join(out_dir, 'train.bin')
    val_bin = os.path.join(out_dir, 'val.bin')
    train_mm = np.memmap(train_bin, dtype=np.uint16, mode='w+', shape=(train_ids.shape[0],))
    train_mm[:] = train_ids[:]
    train_mm.flush()
    val_mm = np.memmap(val_bin, dtype=np.uint16, mode='w+', shape=(val_ids.shape[0],))
    val_mm[:] = val_ids[:]
    val_mm.flush()

    print(f'Wrote {train_bin} ({train_ids.shape[0]} tokens), {val_bin} ({val_ids.shape[0]} tokens)')

if __name__ == '__main__':
    main()
