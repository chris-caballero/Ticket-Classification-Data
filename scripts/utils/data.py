from torch.utils.data import Dataset, DataLoader, random_split

class TicketDataset(Dataset):
    """Custom PyTorch Dataset for support ticket classification."""

    def __init__(self, data, tokenizer, field='complaint_nouns', block_size=200):
        """
        Initialize the TicketDataset.

        Args:
            data (pd.DataFrame): The dataset containing ticket data.
            tokenizer: The tokenizer used to tokenize the text.
            field (str): The field in the dataset containing text data.
            block_size (int): Maximum sequence length for tokenization.
        """
        self.data = data
        self.field = field
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels.
        """
        text = self.data[self.field].iloc[idx]
        label = self.data['label'].iloc[idx]

        # Tokenize the text and create input_ids and attention_mask tensors
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.block_size,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }



def create_dataloader(dataset, batch_size=16):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset: The dataset to create the DataLoader from.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: The created DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True
    )



def to_dataloader(dataset, batch_size=16, split=0.8):
    """
    Convert a dataset to DataLoader(s) with optional train-test split.

    Args:
        dataset: The dataset to convert to DataLoader(s).
        batch_size (int): Batch size for the DataLoader(s).
        split (float): Train-test split ratio (0.0 to 1.0).

    Returns:
        DataLoader or tuple of DataLoaders: The created DataLoader(s).
    """
    if split >= 1 or split <= 0:
        return create_dataloader(dataset, batch_size=batch_size)

    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train = create_dataloader(train_dataset, batch_size=batch_size)
    test = create_dataloader(test_dataset, batch_size=batch_size)

    return train, test
