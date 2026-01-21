from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def translate_sentence(text, model, tokenizer, max_length=128, num_beams=4):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        translated_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def main():
    print("Loading Helsinki-NLP/opus-mt-en-de model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    print("Model and tokenizer loaded.\n")
    test_sentences = ["Language models are fascinating."]
    for sentence in test_sentences:
        print("English:", sentence)
        german = translate_sentence(sentence, model, tokenizer)
        print("German: ", german)
        print("-" * 60)

if __name__ == "__main__":
    main()

