from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from tqdm import tqdm


HOME = '/home/jihyeon/Desktop/Garbage/'
data_dir = os.path.join(HOME, 'data')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 16

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(448),
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

class_names = image_datasets['train'].classes

model_ft = models.resnet50(pretrained=True)
# for param in model_ft.parameters():
#    param.requires_grad = False  # freeze

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs

epoch = 300
#exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, epoch)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                         num_epochs=epoch, save_path='model/resnet50')

model_ft.load_state_dict(torch.load(os.path.join(HOME, 'model/model.pth'), map_location='cuda:1'))

data_transforms = {
    'target': transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
x = 'target'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=4)}

model_ft.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['target']):
        inputs = inputs.to(device)

        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for j in range(inputs.size()[0]):
            result = preds[j]
            print(class_names[result.item()])
            # print(result)