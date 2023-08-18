import torch
import logging
import pandas as pd
import multiprocessing
from transformers import AutoTokenizer
from utils.data import TicketDataset, to_dataloader
from utils.models import EncoderTransformer
from utils.model_utils import evaluate

def setup_logging(log_filename):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename
    )

def load_tickets_dataset(file_path):
    return pd.read_pickle(file_path)

def main():
    setup_logging('logs/evaluate_log')

    # FILES
    TICKETS_FILE = '../data/preprocessed_labeled_complaints.pkl'
    # MODEL_FILE = '../trained_models/text_classification_model.pth'
    MODEL_FILE = 'best_model.pth'

    # HYPER-PARAMETERS
    block_size = 200
    num_classes = 5
    embedding_dim = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Loading Tickets Dataset')

    # Loads the processed and labeled support tickets from file
    complaints = load_tickets_dataset(TICKETS_FILE)

    # Sets up the DataLoaders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = TicketDataset(complaints, tokenizer)
    dataloader = to_dataloader(dataset, split=1)
    vocabulary_size = len(dataset.tokenizer.vocab)

    logging.info('Done!')

    logging.info('Loading Models')

    model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_FILE))

    logging.info('Done!')
    logging.info('Evaluating Models')
    
    accuracy = evaluate(model, dataloader, device=device)

    print('-' * 50 + '\nTRANSFORMER\n' + '-' * 50)
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
