import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

from torchvision.models import resnet18, resnet50, resnet101, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights

import multiprocessing
from torch import autocast



# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='imagenet1K', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms, data_transforms_training, mixup_criterion, mixup_data 



train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms_training),
    batch_size=args.batch_size, shuffle=True, num_workers=4 ,persistent_workers=True,
    pin_memory=False, )
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True,
    pin_memory=False, )

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net, BirdClassifier, LayerResidual
# model = BirdClassifier(20)
#model.load_state_dict(torch.load("experiment/model_117.pth", map_location=torch.device('mps')))

#model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
# model = resnet101(weights=None)

# model.fc = nn.Linear(model.fc.in_features, 20)
# print(model)
# model.load_state_dict(torch.load("experiment/model_200.pth", map_location=torch.device('mps')))
# print(model)
# for name, param in model.named_parameters():
#     if 'fc' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# model2 = Net()
# model2.load_state_dict(torch.load("experiment/model_my_architecture.pth", map_location=torch.device('mps')))

# model3 = BirdClassifier(20)
# model3.load_state_dict(torch.load("experiment/model_best_bird_classifier.pth", map_location=torch.device('mps')))

model = LayerResidual()


model.fc = nn.Linear(model.fc.in_features, 1000)
model.load_state_dict(torch.load("experiment/model_1.pth", map_location=torch.device('mps')))
#restarts at 7

# def reset_batchnorm_stats(model):
#     for module in model.modules():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             module.reset_running_stats()
#             if module.affine:
#                 torch.nn.init.ones_(module.weight)
#                 torch.nn.init.zeros_(module.bias)
# reset_batchnorm_stats(model)


# def check_weights(model):
#     for name, param in model.named_parameters():
#         if torch.isnan(param).any():
#             print(f"❌ NaNs in: {name}")
#         if torch.isinf(param).any():
#             print(f"❌ Infs in: {name}")

# check_weights(model)

# def check_weight_norms(model):
#     for name, param in model.named_parameters():
#         max_val = param.data.abs().max().item()
#         if max_val > 3:  # threshold is generous
#             print(f"⚠️ Unusually large weights in {name}: {max_val}")

# check_weight_norms(model)


# raise NotImplementedError("Please train your own model and load it here.")
# for name, param in model.named_parameters():
#     if 'fc' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)
# model1.to(device)
# model2.to(device)
# model3.to(device)




# optimizer = optim.SGD(model.parameters(), lr=args.lr/10, momentum=args.momentum, weight_decay=1e-4)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr/10,
    weight_decay=1e-4
)

# add cosine annealing learning rate scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=args.epochs
# )

criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        #mixed_data, target_a, target_b, lam = mixup_data(data, target)
        output = model(data)
        
        #output = model(mixed_data)
        #loss = mixup_criterion(criterion, output, target_a, target_b, lam )
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    # scheduler.step()

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            elif use_mps:
                data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss()
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            #print(target[:10])

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    
def validation_soft_voting():
    model1.eval()
    model2.eval()
    model3.eval()

    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        elif use_mps:
            data, target = data.to(device), target.to(device)

        output1 = model1(data).softmax(dim=1)
        output2 = model2(data).softmax(dim=1)
        output3 = model3(data).softmax(dim=1)

        
        pred1 = output1.argmax(dim=1)
        pred2 = output2.argmax(dim=1)
        pred3 = output3.argmax(dim=1)

        preds = torch.stack([pred1, pred2, pred3], dim=1)  # shape: [batch_size, 3]

        final_preds = []
        for i in range(preds.size(0)):
            votes = preds[i]  # tensor of 3 predictions for this sample
            values, counts = votes.unique(return_counts=True)
            
            if counts.max() == 1:  # all different (no majority)
                final_preds.append(pred1[i].item())
            else:
                majority_class = values[counts.argmax()].item()
                final_preds.append(majority_class)

        final_preds = torch.tensor(final_preds, device=preds.device)

        # get the index of the max log-probability
        correct += final_preds.eq(target.data.view_as(final_preds)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validation()
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
