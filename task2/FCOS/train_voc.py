import os
import random
import math, time
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import torch
from dataset.VOC_dataset import VOCDataset
from model.fcos import FCOSDetector

from dataset.augment import Transforms
import numpy as np

import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter


BATCH_SIZE=2
EPOCHS=25

GLOBAL_STEPS=1
LR_INIT= 0.005

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
train_dataset = VOCDataset(root_dir='data/VOCdevkit/VOC2007',resize_size=[800,1333],
                           split='train',use_difficult=False,is_train=True,augment=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=4, worker_init_fn=np.random.seed(0))
    
model = FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.33)

writer = SummaryWriter("logs")

model.train()
for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        lr = optimizer.param_groups[0]['lr']
        print(
            "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
             losses[2].mean(), cost_time, lr, loss.mean()))
            
        GLOBAL_STEPS += 1

    writer.add_scalar('training/total_loss', loss.mean(), epoch + 1)
    writer.add_scalar('training/classification', losses[0].mean(), epoch + 1)
    writer.add_scalar('training/bbox_ctrness', losses[1].mean(), epoch + 1)
    writer.add_scalar('training/bbox_regression', losses[2].mean(), epoch + 1)
    torch.save(model.state_dict(),
               "save_weights/fcos-{}.pth".format(epoch + 1))
    
    if epoch > 3:
        lr_scheduler.step()
    
writer.close()