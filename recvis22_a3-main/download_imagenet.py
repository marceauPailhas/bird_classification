from datasets import load_dataset

from data import HFStreamDataset, data_transforms, data_transforms_training
import torch

#Dataloader for streaming imagenet dataset
train = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
val   = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
stream_ds_train = HFStreamDataset(train, data_transforms_training)
stream_ds_val = HFStreamDataset(val, data_transforms)
train_loader = torch.utils.data.DataLoader(stream_ds_train, batch_size=4, num_workers=0)
val_loader = torch.utils.data.DataLoader(stream_ds_val, batch_size=4, num_workers=0)


for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print("Data shape:", data.shape)      # [batch_size, 3, 224, 224]
    print("Targets:", target)
    
    # your training step here:
    # optimizer.zero_grad()
    # output = model(data)
    # loss = criterion(output, target)
    # loss.backward()
    # optimizer.step()

    if batch_idx == 5:
        break   # just test first few batches