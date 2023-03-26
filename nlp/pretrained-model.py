# from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer

# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# save_dir = "./models/roberta-base"
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# save the model
save_dir = "./models/financial-bert"
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
