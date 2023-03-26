"""Module providing sentiment_analysis"""
from classifier import Classifier
from typing import List

class NLP:
  def __init__(self):
    self.classifier = Classifier()
  def sentiment_analysis(self, text:str): 
    sentiment = self.classifier.get_sentiment_label_and_score(text)
    return sentiment 
  def financial_analysis(self, sentences:List[str]): 
    financial = self.classifier.piping(sentences)
    return financial