import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

def count_parameters(model):
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def get_class_weights(y_true, num_classes, mode=1):
    """
    Calculate class weights based on class frequencies.

    Args:
        y_true (torch.Tensor): True labels.
        num_classes (int): Number of classes.
        mode (int): Weight calculation mode.

    Returns:
        torch.Tensor: Computed class weights.
    """
    total_samples = len(y_true)
    class_frequencies = torch.bincount(y_true, minlength=num_classes)
    
    if mode == 1:
        class_weights = class_frequencies / total_samples
    elif mode == 2:
        class_weights = total_samples / (len(class_frequencies) * class_frequencies)
    else:
        class_weights = None
    
    return class_weights

def train(model, dataloader, is_bert=False, class_weights=None, device=torch.device('cpu'), epochs=5, verbose=True):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for training data.
        is_bert (bool): Whether the model is a BERT-based model.
        class_weights (torch.Tensor): Class weights for loss calculation.
        device (torch.device): Device for computation.
        epochs (int): Number of training epochs.
        verbose (bool): Print training progress.

    Returns:
        nn.Module: Trained PyTorch model.
    """
    criterion = nn.CrossEntropyLoss() if class_weights is None else nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adamax(model.parameters())
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            # Forward pass
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if is_bert:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids)

            # Convert true labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=5).type(torch.float32).to(device)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f'Epoch {epoch} Complete\n- Loss: {loss.item()}')
        
    return model

def evaluate(model, dataloader, is_bert=False, device=torch.device('cpu')):
    """
    Evaluate a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for evaluation data.
        is_bert (bool): Whether the model is a BERT-based model.
        device (torch.device): Device for computation.

    Returns:
        float: Accuracy of the model on the evaluation data.
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if is_bert:
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids)
            
            predictions = torch.argmax(outputs, dim=1).to(device)

            correct += (predictions == labels).sum().item()
            total += len(labels)
            
    accuracy = correct / total
         
    return accuracy

def train_and_evaluate(model=None, training_data=None, validation_data=None, is_bert=False, class_weights=None, epochs=5, verbose=True, device=torch.device('cpu')):
    """
    Train and evaluate a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        training_data (DataLoader): DataLoader for training data.
        validation_data (DataLoader): DataLoader for validation data.
        is_bert (bool): Whether the model is a BERT-based model.
        class_weights (torch.Tensor): Class weights for loss calculation.
        epochs (int): Number of training epochs.
        verbose (bool): Print training progress.
        device (torch.device): Device for computation.

    Returns:
        Tuple[nn.Module, float, float]: Trained model, training accuracy, test accuracy.
    """
    print('Starting Training')
    model = train(model, training_data, is_bert=is_bert, class_weights=class_weights, device=device, epochs=epochs, verbose=verbose)
    
    print('Starting Evaluation')
    train_acc = evaluate(model, training_data, is_bert=is_bert, device=device)
    test_acc = evaluate(model, validation_data, is_bert=is_bert, device=device)
    print(f'Train Accuracy: {train_acc}')
    print(f'Test Accuracy: {test_acc}')

    return model, train_acc, test_acc

class ModelCrossValidation:
    def __init__(self, model, batch_size=32, epochs=5, is_bert=False, class_weights=None, device=torch.device('cpu'), num_splits=10, verbose=True):
        """
        Initialize ModelCrossValidation instance.

        Args:
            model (nn.Module): The PyTorch model.
            batch_size (int): Batch size for data loading.
            epochs (int): Number of training epochs.
            is_bert (bool): Whether the model is a BERT-based model.
            class_weights (torch.Tensor): Class weights for loss calculation.
            device (torch.device): Device for computation.
            num_splits (int): Number of cross-validation splits.
            verbose (bool): Print progress information.
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_bert = is_bert
        self.class_weights = class_weights
        self.device = device
        self.num_splits = num_splits
        self.verbose = verbose

    def run_cross_validation(self, dataset):
        """
        Run cross-validation on the provided dataset.

        Args:
            dataset (Dataset): The dataset for cross-validation.

        Returns:
            Tuple[List[float], float]: List of fold accuracies, cross-validation score.
        """
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=42)
        scores = []

        init_model = deepcopy(self.model)
        # Iterate over each fold
        for i, (train_index, test_index) in enumerate(kf.split(dataset)):
            model = deepcopy(init_model)
            
            # Split the data into training and test sets for this fold
            train_sampler = SubsetRandomSampler(train_index)
            test_sampler = SubsetRandomSampler(test_index)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler)
            
            # Train the model on the training set
            model = train(model, train_loader, is_bert=self.is_bert, class_weights=self.class_weights, device=self.device, epochs=self.epochs, verbose=self.verbose)
            
            # Evaluate the model on the test set using your evaluation function
            score = evaluate(model, test_loader, is_bert=self.is_bert, device=self.device)
            print(f"Fold {i+1} Accuracy: {score}")
            
            # Add the score for this fold to the list of scores
            scores.append(score)
        
        cross_val_score = sum(scores) / self.num_splits

        return scores, cross_val_score
