import torch
from mini_llm.training.loss import calc_loss_loader

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() #1
    with torch.no_grad(): #2
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss