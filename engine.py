import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def loss_fn(outputs, targets):
  return nn.CrossEntropyLoss()(outputs, targets).to(device)

def train_fn(data_loader, model, optimizer, device, scheduler, n_exp):
    model.train()

    correct_predictions = 0
    losses = []

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"].to(device)
        mask = d["mask"].to(device)
        targets = d["targets"].to(device)

        #ids = ids.to(device, dtype=torch.long)
        #mask = mask.to(device, dtype=torch.long)
        #targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)
 
        _, indices = torch.max(outputs, dim=1)  

        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(indices == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / n_exp, np.mean(losses)

def eval_fn(data_loader, model, device, n_exp):
    model.eval()
    
    losses = []
    correct_predictions = 0
   
    with torch.no_grad():    
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"].to(device)
            mask = d["mask"].to(device)
            targets = d["targets"].to(device)

            #ids = ids.to(device, dtype=torch.long)
            #mask = mask.to(device, dtype=torch.long)
            #targets = targets.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask) 
            _, indices = torch.max(outputs, dim=1)  

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            
            correct_predictions += torch.sum(indices == targets)
            
    return correct_predictions.double() / n_exp, np.mean(losses)