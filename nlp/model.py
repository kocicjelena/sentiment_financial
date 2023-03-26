from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer

class Model:
  """A model class to lead the model and tokenizer"""

  def __init__(self) -> None:
    pass
  
  def load_twit_model():
    model_twit = AutoModelForSequenceClassification.from_pretrained("./models/roberta-base/")
    return model_twit

  def load_twit_tokenizer():
    tokenizer_twit = AutoTokenizer.from_pretrained("./models/roberta-base/")
    return tokenizer_twit
  
  def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./models/financial-bert/")
    return model

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("./models/financial-bert/")
    return tokenizer