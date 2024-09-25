from datasets import load_dataset

def prepare_dataset(tokenizer, dataset_name='imdb'):
    # Load the IMDB dataset
    dataset = load_dataset('imdb')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Dataset tokenized successfully!")
    return tokenized_dataset
