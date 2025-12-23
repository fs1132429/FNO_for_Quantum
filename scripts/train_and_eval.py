import torch
import os
import sys
import pickle
import time as t
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Append parent directory to Python path
relative_path = '..'
parent_directory = os.path.abspath(os.path.join(os.getcwd(), relative_path))
sys.path.append(parent_directory)

# Import your modules
from functions.fno_wrapper import FNO1dWrapper as FNO1d
from neuralop import Trainer, CheckpointCallback
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.training import BasicLoggerCallback
from functions.functions import *
from functions.functions_time_model import *
from functions.function_pauli_strings_new_inputs import *
from functions.functions_cuda import *


def mse_loss(y_pred, y, **kwargs):
    assert y_pred.shape == y.shape
    assert y_pred.ndim == 3
    diff = (y_pred - y)
    loss = (diff * diff.conj()).mean(dim=-1).sum(dim=[0, 1])
    return loss.real


def evaluate_model(model_path, data_path, num_states, train_ratio, batch_size,
                   input_T, output_T, thresholds, results_dir, wandb_project_name):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path} ...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model.eval()

    # Load test data
    print("Preparing data loader...")
    _, test_loader = data_preprocess2(data_path, input_T, output_T, num_states, train_ratio, batch_size, device)

    # Run inference
    print("Running inference...")
    predictions_list, ground_truth_list = [], []
    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        with torch.no_grad():
            predictions = model(x)
        predictions_list.append(predictions.cpu())
        ground_truth_list.append(y.cpu())

    predictions = torch.cat(predictions_list, dim=0)
    ground_truth = torch.cat(ground_truth_list, dim=0)

    predictions = predictions[:, :, input_T - output_T:input_T]
    ground_truth = ground_truth[:, :, input_T - output_T:input_T]

    # Plot comparison for a single sample
    fig1, fig2 = plot_comparison_with_error_pauli_strings2(predictions[100, :, 6], ground_truth[100, :, 6])
    comp_path = os.path.join(results_dir, 'comparison_sample.png')
    err_path = os.path.join(results_dir, 'error_sample.png')
    fig1.savefig(comp_path)
    fig2.savefig(err_path)
    plt.close(fig1)
    plt.close(fig2)
    print(f"Saved sample comparison to: {comp_path}")
    print(f"Saved sample error plot to: {err_path}")

    # ---- Evaluation for multiple thresholds ----
    for num, threshold in enumerate(thresholds, start=1):
        print(f"\n===== Evaluation #{num} with threshold = {threshold:.1e} =====")

        # Print fraction above various thresholds for reference
        for t in [8e-4, 7e-4, 9e-4, 5e-4, 1e-3]:
            frac = torch.mean((torch.abs(ground_truth) > t).float())
            print(f"Threshold {t:.0e} → Fraction above: {frac.item():.3f}")

        print("Filtering by threshold and calculating errors...")
        gt_val, pred_val, rel_error, mse_error = filter_by_thresh_2(ground_truth, predictions, threshold)

        rel_error_mean = [np.nanmean(rel) for rel in rel_error]
        mse_error_mean = [np.nanmean(mse) for mse in mse_error]

        # Plot error metrics
        print("Plotting error metrics...")
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(rel_error_mean, label='Mean Relative Error', marker='d')
        axs[0].set_xlabel('Time steps')
        axs[0].set_ylabel('Relative Error')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(mse_error_mean, label='MSE Error', marker='d')
        axs[1].set_xlabel('Time steps')
        axs[1].set_ylabel('MSE Error')
        axs[1].grid(True)
        axs[1].legend()

        err_plot_path = os.path.join(results_dir, f'mean_errors_over_time_{num}.png')
        fig.savefig(err_plot_path)
        plt.close(fig)
        print(f"Saved mean error plot to: {err_plot_path}")

        # Save summary text
        mean_rel_error_val = np.nanmean(rel_error_mean)
        mean_mse_error_val = np.nanmean(mse_error_mean)
        summary_txt = (
            f"Mean Relative Error (averaged over time): {mean_rel_error_val:.6e}\n"
            f"Mean MSE Error (averaged over time): {mean_mse_error_val:.6e}\n"
            f"Threshold: {threshold}\n"
        )
        summary_path = os.path.join(results_dir, f'errors_summary_{num}.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_txt)

        print("\n===== Summary =====")
        print(summary_txt)
        print(f"Saved summary to: {summary_path}")

        # W&B logging
        wandb_eval_run = wandb.init(
            project=f"{wandb_project_name}_eval",
            name=f"eval_run_{num}_{int(t.time())}",
            config={"threshold": threshold, "batch_size": batch_size, "input_T": input_T, "output_T": output_T}
        )

        wandb_eval_run.log({
            "sample_comparison": wandb.Image(comp_path),
            "sample_error": wandb.Image(err_path),
            "mean_error_plot": wandb.Image(err_plot_path),
            "mean_relative_error": mean_rel_error_val,
            "mean_mse_error": mean_mse_error_val,
        })

        wandb_eval_run.finish()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1️⃣ Load data
    train_loader, test_loader = data_preprocess2(
        args.data_path, args.input_T, args.output_T,
        args.num_states, args.train_ratio, args.batch_size, device
    )
    print("Data loaded and preprocessed.")

    # 2️⃣ Define model
    model = FNO1d(
        n_modes_height=args.modes,
        hidden_channels=args.hidden_channels,
        in_channels=args.length + 2,
        out_channels=args.length,
        lifting_channels=args.proj_lift_channel,
        projection_channels=args.proj_lift_channel,
        arch_no=args.arch,
        factorization=args.factorization,
        rank=args.rank
    ).to(device)
    print("Model defined.")

    # 3️⃣ Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, patience=5)
    print("Optimizer and scheduler set.")

    # 4️⃣ Loss functions
    l2loss = LpLoss(d=1, reduce_dims=[0, 1], p=2, reductions=['sum', 'mean'])
    h1loss = H1Loss(d=1, reduce_dims=[0, 1], reductions=['sum', 'mean'])
    eval_losses = {'l2': l2loss, 'h1': h1loss, 'mse': mse_loss}
    print("Loss functions defined.")

    # 5️⃣ W&B setup
    project_name = f"FNO_{args.N}_{args.num_states}_obs_{args.length}_{args.jz}_{args.h}"
    checkpoint_name = (
    f"{args.N}_{int(args.num_states * args.train_ratio)}_"
    f"lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}_gamma{args.gamma}_"
    f"bs{args.batch_size}_hid{args.hidden_channels}_lift{args.proj_lift_channel}_"
    f"modes{args.modes}_rank{args.rank}_{args.scheduler_type}"
)
    save_dir = os.path.join('./checkpoints', project_name, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("/tmp/wandb", exist_ok=True)
    os.chmod("/tmp/wandb", 0o777)
    os.environ["WANDB_DIR"] = "/tmp/wandb"
    wandb_run = wandb.init(project=project_name, name=checkpoint_name, config=vars(args))
    print("W&B initialized.")

    # 6️⃣ Trainer
    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        callbacks=[BasicLoggerCallback()],
        wandb_log=True,
        verbose=True
    )

    # 7️⃣ Train (fixed regularizer argument)
    trainer.train(
        train_loader=train_loader,
        test_loaders={"test_loader": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,      # ✅ Fix: explicitly provide regularizer
        training_loss=l2loss,
        eval_losses=eval_losses,
    )
    print("Training completed.")

    # 8️⃣ Save model
    model_save_path = os.path.join(save_dir, f"model_{args.lr}_{args.gamma}_{args.epochs}.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_save_path}")

    wandb_run.finish()

    # 9️⃣ Evaluation
    results_dir = os.path.join('./results', project_name, checkpoint_name)
    os.makedirs(results_dir, exist_ok=True)
    evaluate_model(
        model_save_path,
        args.data_path,
        args.num_states,
        args.train_ratio,
        args.batch_size,
        args.input_T,
        args.output_T,
        thresholds=[5e-4, 7e-4],
        results_dir=results_dir,
        wandb_project_name=project_name
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--num_states", type=int, default=25000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--input_T", type=int, default=15)
    parser.add_argument("--output_T", type=int, default=10)
    parser.add_argument("--proj_lift_channel", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--arch", type=int, default=3)
    parser.add_argument("--factorization", type=str, default="tucker")
    parser.add_argument("--rank", type=float, default=0.5)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--scheduler_type", type=str, default="ReduceLR", choices=["StepLR", "ReduceLR"])
    parser.add_argument("--jz", type=float, default=0.2)
    parser.add_argument("--h", type=float, default=0.9)
    args = parser.parse_args()

    main(args)
