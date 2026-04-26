import os
from datasets import load_dataset

# Define the target directory matching utils.py
target_dir = "data/wikitext/wikitext-2-raw-v1"
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading wikitext-2-raw-v1 to {target_dir}...")

# Download the dataset from Hugging Face
try:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Save the splits as raw text files
    for split, filename in [("train", "wiki.train.raw"), ("validation", "wiki.valid.raw"), ("test", "wiki.test.raw")]:
        file_path = os.path.join(target_dir, filename)
        print(f"Saving {split} split to {file_path}...")
        with open(file_path, "w", encoding="utf-8") as f:
            # The dataset contains a list of strings, join them
            text = "\n".join(dataset[split]["text"])
            f.write(text)
            
    print("Download complete.")
    print(f"Files saved to {target_dir}")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
