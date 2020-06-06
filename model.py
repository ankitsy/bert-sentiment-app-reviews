import config
import transformers
import torch.nn as nn

class BERTBaseCased(nn.Module):
    """ Returns model"""
    def __init__(self):
        """ Define Model Architecture """
        super(BERTBaseCased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)  # If BERT model stored locally
        #self.bert = transformers.BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # Downloads BERT from transformers
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 3)
        
    def forward(self, ids, mask):
        """ Implements the forward pass """
        last_hidden_state, pooler_output = self.bert(
            ids, 
            attention_mask=mask, 
        )
        output = self.drop(pooler_output)
        output = self.out(output)
        return output