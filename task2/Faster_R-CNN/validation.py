import os
import json
import pandas as pd

import torch
import torchvision
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

import transforms
from networks import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
from voc_dataset import VOCDataSet
from train_utils import ConfusionMatrix

def create_model_faster_rcnn(num_classes):
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def plot_confusion_matrix(mat, names):
    array = mat / (mat.sum(0).reshape(1, 20) + 1E-6)  # normalize
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0)  # for label size
    labels = (0 < len(names) < 99)   # apply names to ticklabels
    sn.heatmap(array, annot=True, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
               xticklabels=names + ['background FN'] if labels else "auto",
               yticklabels=names + ['background FP'] if labels else "auto").set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')



if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = create_model_faster_rcnn(num_classes=21)
    model.to(device)

    acc_lst = []
    miou_lst = []

    writer = SummaryWriter("logs/fr_val")
    for epoch in range(25):
        # load train weights
        weights_path = f"./save_weights/faster-rcnn-{epoch}.pth"
        assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
        weights_dict = torch.load(weights_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        model.load_state_dict(weights_dict)
        model.to(device)

        val_dataset = VOCDataSet("./dataset", "2007", transforms.Compose([transforms.ToTensor()]), "val.txt")
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=0,
                                                      collate_fn=val_dataset.collate_fn)

        conf_mat = ConfusionMatrix(num_classes=20, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5)
        model.eval()
        for i, [images, targets] in enumerate(val_data_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            de = torch.cat((outputs[0]['boxes'],
                            outputs[0]['scores'].unsqueeze(1), outputs[0]['labels'].unsqueeze(1)), 1).detach().cpu().numpy()
            lab = torch.cat((targets[0]['labels'].unsqueeze(1), targets[0]['boxes']), 1).detach().cpu().numpy()
            conf_mat.process_batch(de, lab)

        mat = conf_mat.return_matrix()
        acc = conf_mat.return_acc()
        miou = conf_mat.return_mIoU()

        acc_lst.append(acc)
        miou_lst.append(miou)

        writer.add_scalar('Acc', acc, epoch + 1)
        writer.add_scalar('mIoU', miou, epoch + 1)

    with open('./pascal_voc_classes.json', 'r') as f:
        class_dict = json.load(f)

    plot_confusion_matrix(mat, names=list(class_dict.keys()))
    plt.savefig('./img_out/faster_rcnn_confusion_matrix.jpg')

    val_criteria = pd.DataFrame({'mIoU': miou_lst, 'acc': acc_lst})
    val_criteria.to_csv('faster_cnn_val.csv')

