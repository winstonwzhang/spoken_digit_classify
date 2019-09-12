import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from utils import data, model


########### Hyper-parameters #####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-3
num_epochs = 15
batch_size = 100

wav_dir = 'recordings/'

########### Load data #####################
db = data.Digit_Dataset(wav_dir)
validation_split = .2
shuffle_dataset = True
random_seed = 42

dataset_size = len(db.labels)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=train_sampler, num_workers=2)
val_dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=valid_sampler, num_workers=2)

tr_losses=[]
losses_it=[]

############## Initialize Weights #######################
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


############## Main Function #######################
def main():
    model = BaseModel().double() # Creating the model
    model.apply(init_weights)

    #if torch.cuda.device_count() > 1:
      #   print("Let's use", torch.cuda.device_count(), "GPUs!")
      #  sba = nn.DataParallel(sba)
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
    # train!
    train(model, criterion, optimizer)


############ Validation function #############
def val(model, criterion):
    with torch.no_grad():
        print('Validation starts !!!')
        running_corrects = 0
        for it,data in enumerate(val_dataloader):
            #if it % 10 == 0:
            #    print(it)
            inp = data['img']
            target = data['label']

            inp = inp.double().to(device)
            target = target.to(device)
            model = model.to(device)

            # ===================forward=====================
            output = model(inp)
            _, preds = torch.max(output, 1)
            running_corrects += (preds == target.data).sum().cpu().numpy()

        acc = running_corrects / len(val_indices)
        print('Accuracy:{}'.format(acc))


############ Training function ###################
def train(model, criterion, optimizer):
    print('Initial validation:')
    val(model, criterion)

    print('Training starts !!!')
    for epoch in range(num_epochs):
        start = time.time()

        loss_it = 0.0

        for it,data in enumerate(train_dataloader):
            #if it % 10 == 0:
            #  print(it)
            inp = data['img']
            target = data['label']

            inp = inp.double().to(device)
            target = target.to(device)
            model = model.to(device)
            # ===================forward=====================
            output = model(inp)
            #print(target)
            loss = criterion(output, target)
            #print(loss1.detach().cpu().numpy(), loss2.detach().numpy())
            loss_it += loss.detach().cpu().numpy()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        if (epoch+1) % 1 == 0:
            duration = time.time() - start
            #val_loss1, val_loss2 = val(sba,criterion)
            print('epoch [{}/{}], Tr loss:{:.4f} Time:{:.4f}'.format(epoch+1, num_epochs, loss.detach().cpu().numpy(), duration))
            #val_losses.append([val_loss1, val_loss2])
            tr_losses.append(loss.detach().cpu().numpy())

            tr_losses_npy = np.array(tr_losses)
            it_losses_npy = np.array(losses_it)
            np.save('Train_it_losses.npy',it_losses_npy)
            np.save('Train_losses.npy',tr_losses_npy)

            val(model, criterion)


if __name__ == "__main__":
    main()