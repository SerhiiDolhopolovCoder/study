import torch
from torch import nn
from torch.utils import data as dataloader
from torchmetrics.classification import MulticlassStatScores
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm


def train_step(model: nn.Module,
               dataloader: dataloader.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: MulticlassStatScores,
               device: torch.device) -> tuple[float, float]:
    """
    Returns
        tuple[str, float, float]: (model_name, loss, accuracy)
    """
    loss_sum = 0
    accuracy_sum = 0
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        accuracy_sum += accuracy_fn(y_pred.argmax(dim=1), y).item()
        
    total_loss = loss_sum / len(dataloader)
    total_accuracy = accuracy_sum / len(dataloader)
    return (model.__class__, total_loss, total_accuracy)

def valid_step(model: nn.Module,
               dataloader: dataloader.DataLoader,
               accuracy_fn: MulticlassStatScores,
               loss_fn: nn.Module,
               device: torch.device) -> tuple[float, float]:
    """
    Returns
        tuple[str, float, float]: (model_name, loss, accuracy)
    """
    loss_sum = 0
    accuracy_sum = 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss_sum += loss.item()
            accuracy_sum += accuracy_fn(y_pred.argmax(dim=1), y).item()
    total_loss = loss_sum / len(dataloader)
    total_accuracy = accuracy_sum / len(dataloader)
    return (model.__class__, total_loss, total_accuracy)

def show_results(train_accuracy: list,
                 train_loss: list,
                 valid_accuracy: list,
                 valid_loss: list,
                 epochs: int) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].plot(range(1, epochs + 1), train_accuracy, label='Train accuracy', color='red')
    axs[0].plot(range(1, epochs + 1), valid_accuracy, label='Valid accuracy', color='blue')
    axs[0].set_title('Train and Valid accuracy')
    axs[0].legend() 

    axs[1].plot(range(1, epochs + 1), train_loss, label='Train loss', color='red')
    axs[1].plot(range(1, epochs + 1), valid_loss, label='Valid loss', color='blue')
    axs[1].set_title('Train and Valid loss')
    axs[1].legend() 
    plt.show()
    

def train(model: nn.Module,
          train_dateloader: dataloader.DataLoader,
          valid_dateloader: dataloader.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_fn: MulticlassStatScores,
          device: torch.device,
          epochs: int) -> tuple[float, float, float, float]:
    """
    Returns:
        tuple[float, float, float, float]: (train_loss, train_accuracy, valid_loss, valid_accuracy)
    """
    model.to(device)
    loss_fn.to(device)
    accuracy_fn.to(device)

    start_timer = timer()
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    for epoch in tqdm(range(epochs), desc='Training'):
        print(f"Epoch {epoch + 1}/{epochs}")
        model_name, loss, accuracy = train_step(model=model, 
                                                dataloader=train_dateloader,
                                                loss_fn=loss_fn, 
                                                optimizer=optimizer,
                                                accuracy_fn=accuracy_fn, 
                                                device=device)
        print(f"{model_name} Train loss: {loss:.4f}, Train accuracy: {accuracy:.4f}")
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        model_name,  loss, accuracy = valid_step(model=model, 
                                                dataloader=valid_dateloader,
                                                loss_fn=loss_fn, 
                                                accuracy_fn=accuracy_fn, 
                                                device=device)
        print(f"{model_name} Valid loss: {loss:.4f}, Valid accuracy: {accuracy:.4f}")
        valid_loss.append(loss)
        valid_accuracy.append(accuracy)
        
    end_timer = timer()
    time = end_timer - start_timer
    print(f"Total training time: {time:.2f} seconds")
    
    show_results(
        train_accuracy,
        train_loss,
        valid_accuracy,
        valid_loss,
        epochs
    )
    return train_loss, train_accuracy, valid_loss, valid_accuracy

def predict(X, model: torch.nn.Module, device: torch.device) -> int:
    model.eval()
    with torch.inference_mode():
        X = torch.unsqueeze(X, dim=0).to(device)
        logit = model(X)    
        prob = torch.softmax(logit.squeeze(), dim=0)
        predict_label = torch.argmax(prob)
    return predict_label
    