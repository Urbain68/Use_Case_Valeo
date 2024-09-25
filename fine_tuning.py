from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from model_selection import load_model_and_tokenizer
from dataset_preparation import prepare_dataset
import os


def fine_tune_model(quantize=False):
    # Check if model already trained and saved
    if os.path.exists('./results/pytorch_model.bin'):
        # Load the model with optional 8-bit quantization
        model = DistilBertForSequenceClassification.from_pretrained('./results')
        tokenizer = DistilBertTokenizer.from_pretrained('./results')
        print("Loaded model from saved state.")
    else:
        model, tokenizer = load_model_and_tokenizer()

        tokenized_dataset = prepare_dataset(tokenizer)


        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_dir='./logs',
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
        )

        # Fine-tune the model
        trainer.train()
        print("Model fine-tuning complete!")
        model.save_pretrained('./results')
        tokenizer.save_pretrained('./results')
    # Apply quantization if requested
    if quantize:
        model = DistilBertForSequenceClassification.from_pretrained('./results', load_in_8bit=True)
        print("Model quantized to 8-bit precision.")

    return model, tokenizer