from cgi import test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
import tqdm
from models import resnet
from tqdm import trange
from models.train_models import train,acc_on_test,dic #可更换不同的train函数
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter





transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761)),
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

lr = 0.1
model = resnet.resnet18(num_classes=100).cuda()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80,120,150], gamma=0.1)

writer = SummaryWriter('./log')


epoch = 200
batchsize = 512
loss,val_acc, test_acc = train(model,optimizer,criterion,batchsize, epoch,trainset,testloader,scheduler)
torch.save(model,'./models/output/baseline_aug.pth')
data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
torch.save(data,'./models/datas/baseline_aug_data.pth')

