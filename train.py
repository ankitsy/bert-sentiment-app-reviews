import config
import dataset
from model import BERTBaseCased
import engine

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import defaultdict 
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
  # Read in CSV
  df = pd.read_csv(config.TRAINING_FILE)
  print('Read In Complete!')

  # Split into Validation
  df_train, df_val = train_test_split(df, test_size=0.1, stratify=df.sentiment.values, random_state=config.RANDOM_SEED)
  df_train = df_train.reset_index(drop=True)
  df_val = df_val.reset_index(drop=True)
  print(df_train.shape, df_val.shape)
  print('Validation Split Complete!')

  # Create Dataset required for BERT Model
  train_dataset = dataset.BERTDataset(df_train.content.values, df_train.sentiment.values)
  val_dataset = dataset.BERTDataset(df_val.content.values, df_val.sentiment.values)

  train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
  val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, num_workers=1)
  print('Dataset for Model Complete!')

  # Define Model and Hyperparameters
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = BERTBaseCased()
  model.to(device)

  num_training_steps = len(train_data_loader) * config.EPOCHS
  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=0)

  # Train the Model, Print Aaccurcay, Save Model
  n_train_exp = len(df_train)
  n_val_exp = len(df_val)

  history = defaultdict(list)
  best_accuracy = 0

  for epoch in range(config.EPOCHS):
    print(f'\n{"#" * 10} Epoch: {epoch+1}/{config.EPOCHS} {"#" * 10}\n')
    train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, n_train_exp)    
    val_acc, val_loss = engine.eval_fn(val_data_loader, model, device, n_val_exp)

    print(f'\nTrain Loss: {train_loss:.4f}        Acc: {train_acc:.4f} \nVal   Loss: {val_loss:.4f}    Val Acc: {val_acc:.4f}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        #!rm -rf /content/model*
        torch.save(model.state_dict(), config.MODEL_PATH)  # f'model/model_{val_acc:0.2f}.bin')
        best_accuracy = val_acc


if __name__=="__main__":
  run()