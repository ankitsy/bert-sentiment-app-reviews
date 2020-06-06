import config
import model
import engine
import json
import time
import flask
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torch.nn import functional as F

app = Flask(__name__)

def get_prediction(sentence):
  tokenizer = config.TOKENIZER
  max_len = config.MAX_LEN
  sentence = str(sentence)
  sentence = " ".join(sentence.split())
  inputs = tokenizer.encode_plus(
      sentence,
      None,
      add_special_tokens=True,
      max_length=max_len,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_token_type_ids=False,
      return_tensors='pt'            
  )

  ids = inputs["input_ids"].to(DEVICE)
  mask = inputs["attention_mask"].to(DEVICE)
  
  outputs = MODEL(ids=ids, mask=mask)
  probs = F.softmax(outputs, dim=1)

  _, preds = torch.max(probs, dim=1)  
  
  preds = preds.detach().cpu().item()
  probs = probs.flatten().detach().cpu().numpy().tolist()
  return config.CLASS_NAMES[preds], probs


@app.route('/predict')
def predict():
    # Input sentence and Predictions 
    sentence = request.args.get("sentence") # Code for getting test review from web page'
    class_name, confidence = get_prediction(sentence)
    
    start_time = time.time()
    # Storing web page outputs
    response = {}
    response["response"] = {
        "sentence": f'{sentence}',
        "negative confidence": '{:.5f}'.format(confidence[0]),
        "neutral confidence": '{:.5f}'.format(confidence[1]),
        "positive confidence": '{:.5f}'.format(confidence[2]),
        "time_taken": '{:.3f} sec'.format(time.time() - start_time)
    }
    return flask.jsonify(response)


if __name__=="__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL = model.BERTBaseCased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))  
    MODEL.eval()
    MODEL.to(DEVICE)

    app.run(debug=True)