import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as tud
import torch.distributed as dist
import numpy as np
import time
import os
import argparse
from apex import amp 
from apex.parallel import DistributedDataParallel

# write the codes of 

# TODO:
# 1. Accelerate by CUDA / GPU (checked)
# 2. Figure out the transforms (checked)
# 3. Write the process into file (checked) nohup  method 



def train_model(model, train_dataloader, loss_fn, optimizer, epoch, use_gpu,start,model_type):
    # model.train()
    # for idx, (data,label) in enumerate(train_dataloader):
    #     output = model(data)
    #     loss = loss_fn(output,label)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if idx % 100 == 0 :
    #         print ("Train Epoch: {}, iteration: {}, loss: {}".format(
    #         epoch,idx,loss.item()))
    model.train() # to check 
    total_loss = 0.
    total_corrects = 0.
    for idx, (inputs, labels) in enumerate(train_dataloader):
        print("Iteration: ",idx)
        if (use_gpu):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            # inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        with amp.scale_loss(loss,optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        # preds = outputs.argmax(dim=1).cuda()
        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(preds.eq(labels))
    epoch_loss = total_loss / len(train_dataloader.dataset)
    epoch_accuracy = total_corrects / len(train_dataloader.dataset)
    print("Epoch:{}, Training Loss:{:.4f}, Traning Acc:{:.4f}".format(epoch, epoch_loss, epoch_accuracy))
    interval = time.time()
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
        'timeConsumed':interval-start
    },"./trained/"+model_type+"_"+str(epoch)+r"_checkpoint.pth.tar")

# fopen

def test_model(model, test_dataloader, loss_fn, use_gpu):
    # model.eval()
    # total_loss = 0 
    # correct = 0 
    # with torch.no_grad():
    #     for idx,(data,label) in enumerate(test_dataloader):
    #         output = model(data)
    #         loss = loss_fn(output,label)
    #         pred = output.argmax(dim=1)
    #         total_loss += loss 
    #         correct += (pred==label).sum()
    #     total_loss /= len(test_dataloader.dataset)
    #     acc  = correct.item()/len(test_dataloader.dataset)
    #     print('Test Loss:{}, Accuracy:{}\n'.format(total_loss,acc))
    #     # fopen
    # return acc
    model.eval()
    total_loss = 0.
    total_corrects = 0.
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            # if (use_gpu):
            #     inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # preds = outputs.argmax(dim=1).cuda()
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds.eq(labels))
    epoch_loss = total_loss / len(test_dataloader.dataset)
    epoch_accuracy = total_corrects / len(test_dataloader.dataset)
    print("Test Loss:{:.4f}, Test Acc:{:.4f}".format(epoch_loss, epoch_accuracy))
    return epoch_accuracy


def main(model_type, num_epochs, lr, momentum):
    dist.init_process_group(backend='nccl')


    print("Model initializing ... ")
    if model_type == 'resnet18':
        model = models.resnet18()
    elif model_type == 'vgg16':
        model = models.vgg16()
    elif model_type == 'densenet121':
        model = models.densenet121()
    else:
        print('Unsupported model type.')
        exit()

    # lr = lr
    # momentum = momentum

    # Multi-Processing 1
    # model = torch.nn.DataParallel(model)

    # Multi-Processing 2 
    # torch.distributed.init_process_group(backend="nccl")
    # the run method should be: python -m torch.distributed.launch main.py

    print("Data preparing ... ")

    # data_dir = '/home/ghostinsh3ll/Documents/datasets/ImageNet/ILSVRC2012'
    # data_dir = '/extra_data/amax/ImageNet'
    data_dir = '/data/amax/ImageNet'

    num_class = 1000
    input_size = 224
    batch_size = 64 #256

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # For train data 
    transforms_1 = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    # For val data 
    transforms_2 = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize
    ])

    print("Data preparing ... ")
    train_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transforms_1)
    train_sampler = tud.distributed.DistributedSampler(train_imgs)
    train_dataloader = tud.DataLoader(train_imgs, batch_size=batch_size,sampler=train_sampler)
    # train_dataloader = tud.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)

    test_imgs = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transforms_2)

    test_dataloader = tud.DataLoader(test_imgs, batch_size=batch_size)

    # model = nn.parallel.DistributedDataParallel(model)
    # model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    use_gpu = torch.cuda.is_available()
    if (use_gpu):
        print("This process is running on GPU")
        model.to('cuda')
        model, optimizer = amp.initialize(model,optimizer)
        model = DistributedDataParallel(model)


        # model = model.cuda()
        # model = model.to('cuda')
    #     loss_fn = loss_fn.cuda()

    torch.cuda.synchronize()
    start = time.time()
    print("Start time is ",start,"\n")

    best_valid_acc = 0
    for epoch in range(num_epochs):
        print("Training ",epoch+1," / ",num_epochs,'\n')
        train_model(model, train_dataloader, loss_fn, optimizer, epoch, use_gpu,start,model_type)
        acc = test_model(model, test_dataloader, loss_fn, use_gpu)
        if acc > best_valid_acc:
            best_valid_acc = acc
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            fname = "./trained/" + now + model_type + r"_best_checkpoint.pth"
            print ("Saving best checkpoint to ",fname)
            torch.save(net.state_dict(), fname)
        if epoch == num_epochs - 1:
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            fname = "./trained/" + now + model_type + r"_last_checkpoint.pth"
            print ("Saving final checkpoint to ",fname)
            torch.save(net.state_dict(), fname)
            # save the structure

    torch.cuda.synchronize()
    end = time.time()
    print("End time is ",end,"\n")
    print("Consumed time for ", model_type, " is ", end - start, ".\n")
    # print(''.format)

# fopen


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
args = parser.parse_args()
print(args.local_rank)

# main function 
lr = 0.01
momentum = 0.5
num_epochs = 50
model_types = ['resnet18', 'vgg16']
# model_types = ['resnet18', 'densenet121', 'vgg16']
for i in range(len(model_types)):
    print("Start training ", model_types[i], " ...")
    main(model_types[i], num_epochs, lr, momentum)

# todo: imagenet dataloader 

# vgg 16 

# resnet 

# densenet
