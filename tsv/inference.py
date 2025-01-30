from argparse import ArgumentParser
from .natvamp import get_model_cls
from .data import MNISTDataModule, FMNISTDataModule
import os
from lightning.pytorch import seed_everything
import torch
import numpy as np
import os
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--log-dir", type=str, default="logs")
parser.add_argument("--load", type=str, default="")
parser.add_argument("--model", type=str, default="")
parser.add_argument("--batch-size", type=int, default=200)
parser.add_argument("--out-dir", type=str, default="")

args = parser.parse_args()

EMBEDDINGS_DIR = "embeddings" if args.out_dir == "" else args.out_dir

if __name__ == "__main__":
    seed_everything(42, workers=True)
    load_path = os.path.join(args.log_dir, args.model, args.load, "checkpoints")
    chkpt_path = None
    if os.path.exists(load_path):
        chkpt_path = os.path.join(load_path, os.listdir(load_path)[0])
        print(f"Loading from {chkpt_path}")
        model = (
            get_model_cls(args.model)
            .load_from_checkpoint(
                chkpt_path, sample_input=torch.zeros((1, 1, 28, 28)), lsdim=16
            )
            .eval()
            .to("cuda")
        )
    else:
        raise ValueError(f"Checkpoint at {load_path} not found.")

    data = FMNISTDataModule(num_workers=31, batch_size=args.batch_size, shuffle=False)
    data.setup("fit")
    train_data = data.train_dataloader()
    embeddings = []
    labels = []
    for batch in tqdm(train_data):
        x, y = batch
        z = model(x.to(model.device))[-1]
        embeddings.append(z.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    print("Concatenating embeddings")
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    save_dir = os.path.join(EMBEDDINGS_DIR, args.model, args.load)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving embeddings to {save_dir}")
    np.save(
        os.path.join(save_dir, "embeddings.npy"),
        embeddings,
    )
    np.save(os.path.join(save_dir, "labels.npy"), labels)
