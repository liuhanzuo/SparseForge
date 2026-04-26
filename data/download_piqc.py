import json
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def format_piqa(example):
    """
    Format PIQA example into instruction tuning format.
    PIQA has: 'goal', 'sol1', 'sol2', 'label' (0 or 1)
    """
    goal = example['goal']
    sol1 = example['sol1']
    sol2 = example['sol2']
    label = example['label']
    
    instruction = "Finish the following sentence with the most plausible continuation."
    input_text = f"Goal: {goal}\n\nOption 1: {sol1}\nOption 2: {sol2}"
    
    # Label is 0 for sol1, 1 for sol2
    output = sol1 if label == 0 else sol2
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/instruct')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Downloading PIQA dataset (assuming PIQC refers to PIQA)...")
    try:
        dataset = load_dataset("piqa", trust_remote_code=True)
    except Exception as e:
        print(f"Error downloading PIQA: {e}")
        print("Please check your internet connection or try to download manually.")
        return

    splits = ['train', 'validation']
    output_files = {
        'train': 'piqc_train.json',
        'validation': 'piqc_val.json'
    }
    
    for split in splits:
        if split not in dataset:
            continue
            
        data = dataset[split]
        output_path = os.path.join(args.output_dir, output_files[split])
        
        print(f"Processing {split} split ({len(data)} examples)...")
        formatted_data = []
        for example in tqdm(data):
            formatted_data.append(format_piqa(example))
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
