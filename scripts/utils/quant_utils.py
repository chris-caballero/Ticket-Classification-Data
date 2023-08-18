import torch

def prepare_data_for_quantization(model, dataloader, config=None, is_bert=False, device=torch.device('cpu')):
    """
    Prepare data and model for quantization-aware training (QAT).

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for data.
        config (QConfig): Quantization configuration.
        is_bert (bool): Whether the model is a BERT-based model.
        device (torch.device): Device for computation.
    """
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if is_bert:
            attention_mask = batch['attention_mask'].to(device)
            model.qconfig = config
            qat_transformer = torch.quantization.prepare_qat(model)
            qat_transformer(input_ids, attention_mask)
        else:
            model(input_ids)

def qat_evaluate(model, dataloader, is_bert=False, device=torch.device('cpu')):
    """
    Evaluate a quantized model using quantization-aware training (QAT).

    Args:
        model (nn.Module): The quantized PyTorch model.
        dataloader (DataLoader): DataLoader for evaluation data.
        is_bert (bool): Whether the model is a BERT-based model.
        device (torch.device): Device for computation.

    Returns:
        float: Accuracy of the quantized model on the evaluation data.
    """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
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

def quantize_model(model, config):
    """
    Quantize a PyTorch model using dynamic quantization.

    Args:
        model (nn.Module): The PyTorch model.
        config (QuantizationConfig): Quantization configuration.

    Returns:
        nn.Module: Quantized PyTorch model.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        config,
        dtype=torch.qint8
    )
    return quantized_model
