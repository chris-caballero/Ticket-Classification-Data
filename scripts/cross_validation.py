import torch
import logging
import pandas as pd
import multiprocessing
from transformers import AutoTokenizer
from utils.data import TicketDataset
from utils.models import EncoderTransformer
from utils.model_utils import ModelCrossValidation

def setup_logging(log_filename):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename
    )

def load_tickets_dataset(file_path):
    complaints = pd.read_pickle(file_path)
    return complaints

def main():
    setup_logging('logs/crossval_log')

    # FILES
    TICKETS_FILE = '../data/preprocessed_labeled_complaints.pkl'
    MODEL_FILE = '../trained_models/text_classification_model.pth'

    # HYPER-PARAMETERS
    epochs = 5
    num_classes = 5
    embedding_dim = 300
    block_size = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Loading Tickets Dataset')

    # Loads the processed and labeled support tickets from file
    complaints = load_tickets_dataset(TICKETS_FILE)

    # Sets up the DataLoaders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = TicketDataset(complaints, tokenizer)
    vocabulary_size = len(dataset.tokenizer.vocab)

    logging.info('Done!')
    logging.info('Loading Models')

    model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_classes).to(device)

    logging.info('Done!')

    logging.info('Evaluating Models')

    validator = ModelCrossValidation(
        model=model,
        batch_size=32,
        epochs=epochs,
        is_bert=False,
        device=device,
        num_splits=5,
        verbose=True
    )

    transformer_scores, transformer_crossval = validator.run_cross_validation(dataset)

    logging.info(f'Transformer Scores: {transformer_scores}')
    logging.info(f'Transformer 10-Fold Cross Validation: {transformer_crossval}')

    print(f'Transformer Scores: {transformer_scores}')
    print(f'Transformer 10-Fold Cross Validation: {transformer_crossval}')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
