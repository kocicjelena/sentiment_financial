from scipy.special import softmax
from model import Model
import numpy as np
from transformers import pipeline

class Classifier:
  def __init__(self):
    self.model = Model.load_model()
    self.tokenizer = Model.load_tokenizer()
    self.twit_model = Model.load_twit_model()
    self.twit_tokenizer = Model.load_twit_tokenizer()

  def piping(self, sentences):
    nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
    financial_result=nlp(sentences)
    return financial_result
  
  def get_sentiment_label_and_score(self, text: str):
    result = {}
    labels = ["negative", "neutral", "positive", "macro avg", "weighted avg"]
    encoded_input = self.twit_tokenizer(text, return_tensors='pt')
    output = self.twit_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result["label"] = str(labels[ranking[0]])
    result["score"] = np.round(float(scores[ranking[0]]), 4)
    return result