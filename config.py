import transformers

MAX_LEN = 150
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
EPOCHS = 10
RANDOM_SEED=42
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
BERT_PATH = 'input/bert_base_cased/'
MODEL_PATH = 'assets/model_state.bin'  
TRAINING_FILE = 'input/reviews.csv'

TOKENIZER = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)