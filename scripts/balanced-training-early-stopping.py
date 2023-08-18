import pandas as pd
import torch
from transformers import AutoTokenizer
from utils.data import TicketDataset, to_dataloader
from utils.models import EncoderTransformer
from utils.model_utils import evaluate

SAVE_MODEL_PATH = 'balanced-text-classifier.pth'

def stratified_sampling(group_df, target_size=2300):
    return group_df.sample(target_size)

def main():
    # Hyperparameters
    epochs = 20
    block_size = 200
    embedding_dim = 300
    num_ticket_classes = 5
    batch_size = 32
    patience = 3

    # Load and preprocess data
    tickets = pd.read_pickle('../data/preprocessed_labeled_complaints.pkl')
    sampled_df = tickets.groupby('label', group_keys=False).apply(stratified_sampling)
    sampled_df.reset_index(drop=True, inplace=True)

    # Load tokenizer and create dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = TicketDataset(sampled_df, tokenizer)
    trainset, testset = to_dataloader(dataset, batch_size=batch_size)
    vocabulary_size = len(tokenizer.vocab)

    # Initialize model, criterion, and optimizer
    device = torch.device('cuda')
    model = EncoderTransformer(
        vocabulary_size, embedding_dim, block_size, num_ticket_classes
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters())

    # Training loop
    print('Starting Training')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in trainset:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)

            # Convert true labels to one-hot encoding
            labels = torch.nn.functional.one_hot(labels, num_classes=num_ticket_classes).type(torch.float32).to(device)

            # Compute loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(trainset)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in testset:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)

                # Convert true labels to one-hot encoding
                labels = torch.nn.functional.one_hot(labels, num_classes=num_ticket_classes).type(torch.float32).to(device)

                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(testset)

            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} as validation loss didn't improve for {patience} epochs.")
                break

            print('-' * 75)

        torch.save(best_model, SAVE_MODEL_PATH)

        # Evaluate the best model
        model.load_state_dict(best_model)
        accuracy = evaluate(model, testset, device=device)

        # Print evaluation results
        print('-' * 50 + '\nTRANSFORMER\n' + '-' * 50)
        print('Accuracy:', accuracy)
        print('Done evaluating')

if __name__ == "__main__":
    main()
