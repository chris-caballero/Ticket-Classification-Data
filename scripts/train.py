import warnings
import os
import torch
import logging
import multiprocessing
import pandas as pd
from transformers import AutoTokenizer
from utils.data import TicketDataset, to_dataloader
from utils.models import EncoderTransformer
from utils.model_utils import train_and_evaluate

def setup_logging(log_filename):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filename
    )

def load_tickets_dataset(file_path):
    return pd.read_pickle(file_path)

def main():
    setup_logging('logs/train_log')

    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    save_models = True
    TICKETS_FILE = '../data/preprocessed_labeled_complaints.pkl'
    TRAINED_MODEL_FILE = '../trained_models/text_classification_model.pth'
    SERVING_MODEL_FILE = '../App/server/models/trained_models/text_classification_model-2.pth'

    # HYPER-PARAMETERS
    epochs = 5
    block_size = 200
    num_classes = 5
    embedding_dim = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Device: {device}')
    logging.info('Loading Tickets Dataset')

    # Loads the processed and labeled support tickets from file
    complaints = load_tickets_dataset(TICKETS_FILE)

    # Load the tokenizer and create the dataset using our TicketDataset class
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = TicketDataset(complaints, tokenizer, field='complaint')
    trainset, testset = to_dataloader(dataset, split=0.8)
    vocabulary_size = len(dataset.tokenizer.vocab)

    logging.info('Done!')

    logging.info('Creating Models')

    # MODEL INITIALIZATION
    model = EncoderTransformer(vocabulary_size, embedding_dim, block_size, num_classes).to(device)

    logging.info('Done!')
    logging.info('Training and Evaluating Models')

    # TRAINING AND EVALUATION BLOCK
    train_and_evaluate(model, trainset, testset, epochs=5, verbose=True, device=device)

    logging.info('Done!')

    # SAVE MODELS
    if save_models:
        logging.info('Saving Models')
        torch.save(model.state_dict(), TRAINED_MODEL_FILE)
        torch.save(model.state_dict(), SERVING_MODEL_FILE)
        logging.info('Done!')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()