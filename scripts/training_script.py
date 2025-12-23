import torch
import os
import sys
# Define the relative path to the parent directory
relative_path = '..'

# Append the parent directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), relative_path))
sys.path.append(parent_directory)
import pickle
import argparse
import time as t
from functions.functions import *
from functions.functions_time_model import *
from functions.function_pauli_strings_new_inputs import *
from functions.functions_cuda import *
import matplotlib.pyplot as plt

import torch.nn.functional as F
from functions.fno_wrapper import FNO1dWrapper as FNO1d
from neuralop import Trainer, CheckpointCallback
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.training import setup, BasicLoggerCallback
import wandb    
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau



def mse_loss(y_pred, y, **kwargs):
    #print(y_pred.shape,y.shape)
    assert y_pred.shape == y.shape
    assert y_pred.ndim == 3
    diff = (y_pred - y) 
    loss = (diff * diff.conj()).mean(dim=-1).sum(dim=[0, 1])
    return loss.real


def main(N, num_states, train_ratio, batch_size, hidden_channels, input_T, output_T,
         proj_lift_channel, epochs, lr, gamma, weight_decay, data_path, step_size,
         arch, factorization, rank, length, modes, scheduler_type, jz, h):

    # Load and preprocess data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = data_preprocess2(data_path, input_T, output_T, num_states, train_ratio, batch_size, device)
    print("data preprocessed")
    # Model definition
    model = FNO1d(
        n_modes_height=modes,
        hidden_channels=hidden_channels,
        in_channels=length + 2,
        out_channels=length,
        lifting_channels=proj_lift_channel,
        projection_channels=proj_lift_channel,
        arch_no=arch,
        factorization=factorization,
        rank=rank
    )
    
    print(f"Using device: {device}")
    model = model.to(device)
    
    print("model defined")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler selection
    if scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ReduceLR":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5)
    else:
        raise ValueError("scheduler_type must be either 'StepLR' or 'ReduceLR'")
    
    print("scheduler defined")

    # Loss functions
    l2loss = LpLoss(d=1, reduce_dims=[0, 1], p=2, reductions=['sum', 'mean'])
    h1loss = H1Loss(d=1, reduce_dims=[0, 1], reductions=['sum', 'mean'])
    eval_losses = {'l2': l2loss, 'h1': h1loss, 'mse': mse_loss}
    training_loss = l2loss
    
    print("loss functions defined")

    project_name = f"FNO_{N}_{num_states}_obs_{length}_{jz}_{h}"
    checkpoint_name = f"{N}_{int(num_states*train_ratio)}_{lr}_{epochs}_{gamma}_{batch_size}_{hidden_channels}_{proj_lift_channel}_{scheduler_type}"
    os.makedirs("/tmp/wandb", exist_ok=True)
    os.chmod("/tmp/wandb", 0o777)
    os.environ["WANDB_DIR"] = "/tmp/wandb"
    run = wandb.init(project=project_name, config={
        "particles": N,
        "batch_size": batch_size,
        "training_data": int(num_states * train_ratio),
        "lr": lr,
        "gamma": gamma,
        "hidden_channels": hidden_channels,
        "proj_lift": proj_lift_channel,
        "epochs": epochs,
        "modes": modes,
        "rank": rank,
        "factorization": factorization,
        "step_size": step_size,
        "scheduler": scheduler_type
    }, name=checkpoint_name)
    
    print("wandb initialized")

    # Save directory
    save_dir = os.path.join('./checkpoints', project_name, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)


    trainer = Trainer(
        model=model,
        n_epochs=epochs,
        device=device,
        callbacks=[BasicLoggerCallback()],
        data_processor=None,
        wandb_log=True,
        log_test_interval=1,
        use_distributed=False,
        verbose=True
    )
    
    print("trainer defined")

    trainer.train(
        train_loader=train_loader,
        test_loaders={"test_loader": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=training_loss,
        eval_losses=eval_losses,
    )
    
    print("training completed")

    # Save model
    model_save_path = os.path.join(save_dir, f"model_{lr}_{gamma}_{epochs}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a FNO model.")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--num_states", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--input_T", type=int, default=20)
    parser.add_argument("--output_T", type=int, default=10)
    parser.add_argument("--proj_lift_channel", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--arch", type=int, default=3)
    parser.add_argument("--factorization", type=str, default="tucker")
    parser.add_argument("--rank", type=float, default=0.5)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--scheduler_type", type=str, default="StepLR", choices=["StepLR", "ReduceLR"])
    parser.add_argument("--jz", type=float, default=1)
    parser.add_argument("--h", type=float, default=1)

    args = parser.parse_args()
    main(**vars(args))
