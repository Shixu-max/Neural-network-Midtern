from statistics import mode
import torch
from tqdm import trange
from random import shuffle
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
from copy import deepcopy
from tensorboardX import SummaryWriter



writer = SummaryWriter()
def train(model,optimizer,criterion,batch_size,epochs,trainset,testloader, scheduler=None):
    total_loss = []
    test_acc = []
    valid_acc = []
    l = len(trainset)
    train_size = int(0.8*l)
    valid_size = l-train_size
    for epoch in trange(epochs):  # loop over the dataset multiple times
        train_, valid_ = torch.utils.data.random_split(trainset, [train_size, valid_size])
        trainloader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True, num_workers=2)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # inputs, labels_a, labels_b = Variable(inputs), Variable(labels_a), Variable(labels_b)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward() # 反向传播，计算参数的更新值

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=10,norm_type=2)

            optimizer.step() # 将计算得到的参数加到Net上
            
            epoch_loss += loss.item()
            
        if scheduler is not None:
            scheduler.step()
                
        print('[%d, %5d] loss: %.3f' %(epoch + 1, epochs, epoch_loss))
        # for name, parms in model.named_parameters():
        #     print('-->name:',name)
        #     print('-->grad_value:',parms.grad)       
        
        total_loss.append(epoch_loss)
        # test
        acc_test = acc_on_test(model,testloader)
        acc_valid = acc_on_test(model,validloader)
        test_acc.append(acc_test)
        valid_acc.append(acc_valid)
    loss_per_epoch = [a / l for a in total_loss]
    print('Finished Training')
    return loss_per_epoch, valid_acc, test_acc
    # return loss_per_epoch,test_acc


def train_v2(model,optimizer,criterion,epochs,trainloader,testloader, scheduler=None):
    total_loss = []
    test_acc = []
    for epoch in trange(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(inputs.shape)
            outputs = model(inputs)
            label = torch.zeros(outputs.size())
            label = label.cuda()
            for j, index in enumerate(labels):
                label[j][index] = 1
            # numbers, outputs = torch.max(outputs,1)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, label)
            # loss = loss.float()
            # loss.requires_grad_(True)
            

            loss.backward() # 反向传播，计算参数的更新值
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameter(),max_norm=5,norm_type=2)
            optimizer.step() # 将计算得到的参数加到Net上
            
            epoch_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        
        print('[%d, %5d] loss: %.3f' %(epoch + 1, epochs, epoch_loss))


        
        total_loss.append(epoch_loss)
        # test
        acc = acc_on_test(model,testloader)
        test_acc.append(acc)
    loss_per_epoch = [a / 50000 for a in total_loss]
    print('Finished Training')
    return loss_per_epoch, test_acc


def train_mixup(model,optimizer,criterion,batch_size,epochs,trainset,testloader, scheduler=None):
    total_loss = []
    test_acc = []
    valid_acc = []
    l = len(trainset)
    train_size = int(0.8*l)
    valid_size = l-train_size
    for epoch in trange(epochs):  # loop over the dataset multiple times
        train_, valid_ = torch.utils.data.random_split(trainset, [train_size, valid_size])
        trainloader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True, num_workers=2)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            inputs, labels_a, labels_b = Variable(inputs), Variable(labels_a), Variable(labels_b)

            outputs = model(inputs)

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, outputs)
            # loss = criterion(outputs, labels)

            loss.backward() # 反向传播，计算参数的更新值

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=10,norm_type=2)

            optimizer.step() # 将计算得到的参数加到Net上
            
            epoch_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
                
        print('[%d, %5d] loss: %.3f' %(epoch + 1, epochs, epoch_loss))

        
        total_loss.append(epoch_loss)
        # test
        acc_test = acc_on_test(model,testloader)
        acc_valid = acc_on_test(model,validloader)
        test_acc.append(acc_test)
        valid_acc.append(acc_valid)
    loss_per_epoch = [a / l for a in total_loss]
    print('Finished Training')
    return loss_per_epoch, valid_acc, test_acc


def train_cutmix(model,optimizer,criterion,batch_size,epochs,trainset,testloader, scheduler=None):
    total_loss = []
    test_acc = []
    valid_acc = []
    l = len(trainset)
    train_size = int(0.8*l)
    valid_size = l-train_size
    for epoch in trange(epochs):  # loop over the dataset multiple times
        train_, valid_ = torch.utils.data.random_split(trainset, [train_size, valid_size])
        trainloader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True, num_workers=2)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, alpha=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            inputs, labels_a, labels_b = Variable(inputs), Variable(labels_a), Variable(labels_b)

            outputs = model(inputs)

            loss_func = cutmix_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, outputs)
            # loss = criterion(outputs, labels)

            loss.backward() # 反向传播，计算参数的更新值

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=10,norm_type=2)

            optimizer.step() # 将计算得到的参数加到Net上
            
            epoch_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
                
        print('[%d, %5d] loss: %.3f' %(epoch + 1, epochs, epoch_loss))

        
        total_loss.append(epoch_loss)
        # test
        acc_test = acc_on_test(model,testloader)
        acc_valid = acc_on_test(model,validloader)
        test_acc.append(acc_test)
        valid_acc.append(acc_valid)
    loss_per_epoch = [a / l for a in total_loss]
    print('Finished Training')
    return loss_per_epoch, valid_acc, test_acc

def train_cutout(model,optimizer,criterion,batch_size,epochs,trainset,testloader, scheduler=None):
    total_loss = []
    test_acc = []
    valid_acc = []
    l = len(trainset)
    train_size = int(0.8*l)
    valid_size = l-train_size
    for epoch in trange(epochs):  # loop over the dataset multiple times
        train_, valid_ = torch.utils.data.random_split(trainset, [train_size, valid_size])
        trainloader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True, num_workers=2)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs, labels, lam = cutout_data(inputs, labels, alpha=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            loss_func = cutout_criterion(labels, lam)
            loss = loss_func(criterion, outputs)
            # loss = criterion(outputs, labels)

            loss.backward() # 反向传播，计算参数的更新值

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=10,norm_type=2)

            optimizer.step() # 将计算得到的参数加到Net上
            writer.add_scalar('cutout_'+"/loss_train", loss.item(), epoch)
            
            
            epoch_loss += loss.item()
            
            
        if scheduler is not None:
            scheduler.step()
                
        print('[%d, %5d] loss: %.3f' %(epoch + 1, epochs, epoch_loss))

        
        total_loss.append(epoch_loss)
        # test
        acc_test = acc_on_test(model,testloader)
        acc_valid = acc_on_test(model,validloader)
        test_acc.append(acc_test)
        valid_acc.append(acc_valid)
    loss_per_epoch = [a / l for a in total_loss]
    print('Finished Training')
    return loss_per_epoch, valid_acc, test_acc


def acc_on_test(model,testloader):
    correct = 0
    total = 0
    for input, target in testloader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        numbers, predic = torch.max(output,1)
        total += target.size(0) # 加一个batch_size
        correct += (predic==target).sum().item()
    acc = correct / total
    epoch = 200
    writer.add_scalar('cutout_'+"/acc_train", 100. * acc, epoch)  
    return acc



def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda() #随机排列

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x,y,alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    
    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices]
    shuffled_y = y[indices]

    image_h, image_w = x.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    x[:, :, y0:y1, x0:x1] = shuffled_x[:, :, y0:y1, x0:x1]
    y_a,y_b = y,shuffled_y

    return x,y_a,y_b,lam

def cutmix_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
def rand_bbox(size, lam):
    s0 = size[0]
    s1 = size[1]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mask = np.ones((s0, s1, W, H), np.float32)
    mask[:, :, bbx1: bbx2, bby1: bby2] = 0.
    mask = torch.from_numpy(mask)
    return mask
    
def cutout_data(x, y, alpha=1.0,use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()


    #y_a, y_b = y, y[index]
    mask = rand_bbox(x.size(), lam)

    mask = mask.to().cuda()
    x = x * mask
    #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y,lam

def cutout_criterion(y, lam):
    return lambda criterion, pred: lam * criterion(pred, y)



def Cutmix(img1,img2):
    h = img1.size(1)
    w = img1.size(2)
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - 16 // 2, 0, h)
    y2 = np.clip(y + 16 // 2, 0, h)
    x1 = np.clip(x - 16 // 2, 0, w)
    x2 = np.clip(x + 16 // 2, 0, w)

    print(x1,x2,y1,y2)

    img3 = deepcopy(img1)
    img3[y1: y2, x1: x2] = img2[y1: y2, x1: x2]

    return img3

dic = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}