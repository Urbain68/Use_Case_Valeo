from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Download and load the model and tokenizer
def load_model_and_tokenizer(model_name='distilbert-base-uncased'):
    # Téléchargement du modèle et du tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer