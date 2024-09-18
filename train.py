"Train script for the model"
import os
from dataclasses import asdict
import json

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from models.patch_attention_unet import Config, PatchAttentionUNET
from models import PatchAttentionUNET_v2, PatchAttentionUNET_v2Config
from dataset import MovingGIFInterpolationDataset

run_id = max(
    [int(dirname)
     for dirname in os.listdir("runs") if dirname.isnumeric()]) + 1

train_dataset = MovingGIFInterpolationDataset(
    "data/moving-gif-processed/moving-gif/train", skip_frames=1)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = MovingGIFInterpolationDataset(
    "data/moving-gif-processed/moving-gif/test", skip_frames=1)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

config = PatchAttentionUNET_v2Config()
model = PatchAttentionUNET_v2(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
test_losses = []

for epoch in range(100):
    model.train()
    for i, batch in enumerate(train_loader):
        if batch is None:
            continue

        (inputs, target) = batch

        inputs = (inputs[0].to(device), inputs[1].to(device))
        target = target.to(device)

        output = model(inputs)

        loss = torch.nn.functional.mse_loss(output, target)

        print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")
        train_losses.append(loss.detach())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if not os.path.exists(f"runs/{run_id}"):
        os.makedirs(f"runs/{run_id}/checkpoints")
        os.makedirs(f"runs/{run_id}/plots")
        with open(f"runs/{run_id}/config.json", "w",
                  encoding="utf-8") as config_file:
            json.dump(asdict(config), config_file, indent=2)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                continue

            (inputs, target) = batch

            inputs = (inputs[0].to(device), inputs[1].to(device))
            target = target.to(device)

            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output, target)

            print(f"[TEST] Epoch {epoch}, Batch {i}, Loss {loss.item()}")
            test_losses.append(loss.detach())

            if i == 0:
                fig, ax = plt.subplots(target.shape[0], 4)
                fig.set_size_inches(16, target.shape[0] * 4)
                for j in range(target.shape[0]):
                    ax[j, 0].imshow(inputs[0][j].detach().squeeze().permute(
                        1, 2, 0))
                    ax[j, 1].imshow(inputs[1][j].detach().squeeze().permute(
                        1, 2, 0))
                    ax[j,
                       2].imshow(target[j].detach().squeeze().permute(1, 2, 0))
                    ax[j, 3].imshow(output[j].clamp(
                        0, 1).detach().squeeze().permute(1, 2, 0))

                fig.savefig(f"runs/{run_id}/plots/results_{epoch}.png")
                fig.clear()

    plt.clf()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot([
        x * (len(train_losses) / len(test_losses))
        for x in range(len(test_losses))
    ], test_losses)
    plt.savefig(f"runs/{run_id}/plots/loss.png")
    plt.close()

    torch.save(model.state_dict(),
               f"runs/{run_id}/checkpoints/model_{epoch}.pth")
