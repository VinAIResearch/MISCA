import os

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, args):
        if args.tuning_metric == "loss":
            score = -val_loss
        else:
            score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args)
        elif score < self.best_score:
            if self.patience > 0:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, args):
        """Saves model when validation loss decreases or accuracy/f1 increases."""
        if self.verbose:
            if args.tuning_metric == "loss":
                print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
            else:
                print(
                    f"{args.tuning_metric} increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
        torch.save(model, os.path.join(args.model_dir, "model.bin"))
        torch.save(args, os.path.join(args.model_dir, "training_args.bin"))
        self.val_loss_min = val_loss