import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd

def plot_losses(logpath, outdir, name, show_only=False):
    """Plots training and validation loss"""
    stats = pd.read_csv(os.path.join(logpath, name + "_training.log")).values
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    axes = axes.ravel()

    epoch = stats[:, 0]
    train_acc = stats[:, 1]
    train_loss = stats[:, 2]
    val_acc = stats[:, 3]
    val_loss = stats[:, 4]

    best_loss = np.argmin(val_loss)

    axes[0].plot(epoch, train_loss, "o-", label="train loss", color="salmon")
    axes[0].plot(epoch, val_loss, "o-", label="val loss", color="lightblue")
    axes[0].axvline(best_loss, color="black", ls="--", alpha=0.5)

    axes[1].plot(
        epoch, train_acc, "o-", label="train acc (best: {:.4f})".format(train_acc.max()), color="salmon"
    )
    axes[1].plot(epoch, val_acc, "o-", label="val acc (best: {:.4f})".format(val_acc.max()), color="lightblue")
    axes[1].axvline(best_loss, color="black", ls="--", alpha=0.5)

    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    if show_only:
        plt.show()
    else:
        plt.savefig(os.path.join(outdir, name + "_loss.pdf"))
        plt.close()
