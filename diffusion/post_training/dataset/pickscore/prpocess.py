from datasets import load_dataset, Dataset
import random

# Load the original dataset
dataset = load_dataset('yuvalkirstain/pickapic_v1', num_proc=16)

# Process train split
text_dataset = dataset['train'].select_columns(["caption"])
unique_dataset = text_dataset.unique("caption")
unique_dataset = [s for s in unique_dataset if s.count(' ') >= 5]

# Shuffle the unique dataset
random.shuffle(unique_dataset)

# Split into test (2048 samples) and train (remaining samples)
test_size = 2048
unique_text_dataset = unique_dataset[:test_size]
train_dataset = unique_dataset[test_size:]

# Save the datasets with shuffling
with open("dataset/pickscore/train.txt", "w", encoding="utf-8") as file:
    for line in train_dataset:
        file.write(line + "\n")

with open("/dataset/pickscore/test.txt", "w", encoding="utf-8") as file:
    for line in unique_text_dataset:
        file.write(line + "\n")