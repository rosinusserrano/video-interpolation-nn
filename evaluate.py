"Evaluate trained models"
import torch
import matplotlib.pyplot as plt
import numpy as pd

from models.patch_attention_unet import Config, PatchAttentionUNET
from dataset import MovingGIFInterpolationDataset

train_dataset = MovingGIFInterpolationDataset(
    "data/moving-gif-processed/moving-gif/train")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=16,
                                           shuffle=True)
test_dataset = MovingGIFInterpolationDataset(
    "data/moving-gif-processed/moving-gif/test")
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False)

config = Config()
model = PatchAttentionUNET(config)
model.load_state_dict(torch.load("runs/1/checkpoints/model_9.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for i, batch in enumerate(test_loader):
    (x, z), y_true = batch
    y_pred = model((x, z))

    y_pred = y_pred.clamp(0, 1)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(x.detach().squeeze().permute(1, 2, 0))
    ax[0, 1].imshow(z.detach().squeeze().permute(1, 2, 0))
    ax[1, 0].imshow(y_true.detach().squeeze().permute(1, 2, 0))
    ax[1, 1].imshow(y_pred.detach().squeeze().permute(1, 2, 0))

    fig.savefig(f"runs/1/eval/{i}.png")

    if i > 10:
        break

# train_losses = []
# test_losses = []

# model.eval()
# with torch.no_grad():
#     for i, batch in enumerate(train_loader):
#         if batch is None:
#             continue

#         (inputs, target) = batch

#         inputs = (inputs[0].to(device), inputs[1].to(device))
#         target = target.to(device)

#         output = model(inputs)
#         loss = torch.nn.functional.mse_loss(output, target)

#         print(f"[TRAIN | Batch {i}] Loss {loss}")
#         train_losses.append(loss)

#     for i, batch in enumerate(test_loader):
#         if batch is None:
#             continue

#         (inputs, target) = batch

#         inputs = (inputs[0].to(device), inputs[1].to(device))
#         target = target.to(device)

#         output = model(inputs)
#         loss = torch.nn.functional.mse_loss(output, target)

#         print(f"[TEST | Batch {i}] Loss {loss}")
#         test_losses.append(loss)

# plt.scatter([1] * len(train_losses), train_losses)
# plt.scatter([2] * len(test_losses), test_losses)
# plt.show()
