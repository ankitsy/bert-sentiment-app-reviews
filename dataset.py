import torch
import config

class BERTDataset:
    """ Return input_ids, attention_masks, token_type_ids, targets """

    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        """ This function returns the number of samples in text """
        return len(self.review)
 
    def __getitem__(self, item):
        """ This function returns the token_ids, attention_mask, and token_type_ids for each index in text """
        review = str(self.review[item])
        review = " ".join(review.split())
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'            
        )
        
        ids = inputs["input_ids"].flatten()
        mask = inputs["attention_mask"].flatten()

        return {
            "ids" : ids,
            "mask" : mask,
            "targets": torch.tensor(self.target[item], dtype=torch.long),           
        }