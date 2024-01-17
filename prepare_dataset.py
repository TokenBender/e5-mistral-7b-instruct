from datasets import load_dataset, Dataset, DatasetDict

dataset = load_dataset("your_dataset_name")  # Replace with the actual dataset name

train_dataset = []

for example in dataset["train"]:
    example["instruction"] = 'Instruct: ' + example["task"] + '\nQuery: ' + example["query"]
    train_dataset.append({
        "instruction": example["instruction"],
        "positive": example["pos"],
        "negative": example["neg"]
    })

data = Dataset.from_list(train_dataset)

train_test_split = data.train_test_split()
test_valid = train_test_split['test'].train_test_split(test_size=0.5)
train_test_valid_dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

print(train_test_valid_dataset)

train_test_valid_dataset.save_to_disk("similarity_dataset/")