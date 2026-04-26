from datasets import load_dataset
import tiktoken
import random
import pickle
import torch
import os

# Get absolute path to the data directory
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data', 'c4_dataset')
json_path = [os.path.join(data_dir, f'c4-train.{i:05d}-of-01024.json.gz') for i in range(0, 17)]
# Keep only files that actually exist
json_path = [p for p in json_path if os.path.exists(p)]
if not json_path:
    raise FileNotFoundError("No C4 files found in data_dir for c4-train.00000-of-01024.json.gz .. c4-train.00016-of-01024.json.gz")
cache_dir = os.path.join(data_dir, 'cache', 'train')
output_path = os.path.join(data_dir, 'calibration_dataset.pkl')

print(f"Loading data from {json_path}")
traindata =  load_dataset('json', data_files={'train': json_path}, 
                            split='train',cache_dir=cache_dir)
enc = tiktoken.get_encoding("gpt2")
seed  = 42
seqlen = 1024

nsamples = 128
trainloader = []
for _ in range(nsamples):
    while True:
        i = random.randint(0, len(traindata) - 1)
        # transform list to tensor

        trainenc = enc.encode_ordinary(traindata[i]['text'])
        trainenc = torch.tensor(trainenc, dtype=torch.long).unsqueeze(0)
        if trainenc.shape[1] > seqlen:
            break
    i = random.randint(0, trainenc.shape[1]  - seqlen - 1)
    j = i + seqlen
    inp = trainenc[:, i:j]
    trainloader.append(inp)

# save the calibration dataset
print(f"Saving calibration dataset to {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(trainloader, f)
