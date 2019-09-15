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
from utils import data
from utils import model as mdl


########### Hyper-parameters #####################
class hp:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 3e-4  # learning rate
    num_epochs = 50
    batch_size = 50
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42
    num_w = 8  # number of parallel workers
    weight_decay = 1e-3

print(hp.device)

wav_dir = 'recordings/'
save_path = 'models/fsdd_cnn_sdict.pt'

########### Load data #####################
train_db = data.Digit_Dataset(wav_dir, transform=True)
val_db = data.Digit_Dataset(wav_dir, transform=False)

dataset_size = len(train_db.labels)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if hp.shuffle_dataset :
    np.random.seed(hp.random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
test_indices = val_indices[:len(val_indices) // 2]
val_indices = val_indices[len(val_indices) // 2:]  # split into val and test

num_train = len(train_indices)
num_val = len(val_indices)
num_test = len(test_indices)

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_dataloader = DataLoader(train_db, batch_size=hp.batch_size, 
                              sampler=train_sampler, num_workers=hp.num_w)
val_dataloader = DataLoader(val_db, batch_size=hp.batch_size,
                            sampler=valid_sampler, num_workers=hp.num_w)
test_dataloader = DataLoader(val_db, batch_size=hp.batch_size,
                            sampler=test_sampler, num_workers=hp.num_w)


############## Initialize Weights #######################
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


############## Main Function #######################
def run():
    model = mdl.BaseModel().double() # Creating the model
    model.apply(init_weights)

    #if torch.cuda.device_count() > 1:
      #   print("Let's use", torch.cuda.device_count(), "GPUs!")
      #  sba = nn.DataParallel(sba)
    model = model.to(hp.device)

    criterion = nn.CrossEntropyLoss().to(hp.device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp.lr,
                                 weight_decay=hp.weight_decay)
        
    # train!
    result = train(model, criterion, optimizer)
    print(result)
    # test
    test_acc, test_loss = val(model, criterion, test_dataloader)
    print('test_acc:{}, test_loss:{}'.format(test_acc, test_loss))


############ Validation function #############
def val(model, criterion, dl):
    with torch.no_grad():
      
        print('Validation starts !!!')
        
        running_corrects = 0
        loss_sum = 0.0
        
        for it, data in enumerate(dl):
            #if it % 10 == 0:
            #    print(it)
            inp = data['img']
            target = data['label']

            inp = inp.double().to(hp.device)
            target = target.to(hp.device)
            model = model.to(hp.device)

            # ===================forward=====================
            output = model(inp)
            loss = criterion(output, target)
            loss_sum += loss.item()
            _, preds = torch.max(output, 1)
            running_corrects += (preds == target.data).sum().cpu().numpy()

        acc = running_corrects / num_val
        loss_sum = loss_sum / len(dl)
        print('Accuracy:{0}, Loss:{1}'.format(acc, loss_sum))
        return acc, loss_sum


############ Training function ###################
def train(model, criterion, optimizer):
    print('Initial validation:')
    val(model, criterion, val_dataloader)

    print('Training starts !!!')
    tr_losses=[]
    tr_accs = []
    val_losses=[]
    val_accs = []
    best_epoch = 0
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        start = time.time()
        
        model.train()  # train mode

        running_corrects = 0
        loss_sum = 0.0

        for it,data in enumerate(train_dataloader):
            #if it % 10 == 0:
            #  print(it)
            inp = data['img']
            target = data['label']

            inp = inp.double().to(hp.device)
            target = target.to(hp.device)
            model = model.to(hp.device)
            # ===================forward=====================
            output = model(inp)
            loss = criterion(output, target)
            loss_sum += loss.item()
            _, preds = torch.max(output, 1)
            running_corrects += (preds == target.data).sum().cpu().numpy()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # eval mode
        
        duration = time.time() - start
        loss_sum = loss_sum / len(train_dataloader)
        tr_acc = running_corrects / num_train
        print('epoch [{}/{}], Tr loss:{:.4f} Tr Acc:{:.4f} Time:{:.4f}'.format(
            epoch+1, hp.num_epochs, loss_sum, tr_acc, duration))
        tr_losses.append(loss_sum)
        tr_accs.append(tr_acc)

        val_acc, val_loss = val(model, criterion, val_dataloader)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
    
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            # save model
            torch.save(model.state_dict(), save_path)
    
    # save accs and loss
    np.save('tr_accs.npy', np.array(tr_accs))
    np.save('val_accs.npy', np.array(val_accs))
    np.save('train_losses.npy', np.array(tr_losses))
    np.save('val_losses.npy', np.array(val_losses))
    
    return {'best_epoch': best_epoch, 'best_acc': best_acc}


if __name__ == "__main__":
    run()
