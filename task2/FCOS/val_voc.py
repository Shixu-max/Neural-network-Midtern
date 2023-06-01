import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.1, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)


        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]


        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[0, gt_class] += 1  # background

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
                detection_class = detection_classes[i]
                self.matrix[detection_class, 0] += 1  # background

    def return_matrix(self):
        return self.matrix[1:21, 1:21] # only return confusion matrix of fg

    def return_acc(self):
        confusion_matrix = self.return_matrix()
        acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0
        return acc

    def return_mIoU(self):
        confusion_matrix = self.return_matrix()
        detection_sum = np.sum(confusion_matrix, axis=0)
        ground_truth_sum = np.sum(confusion_matrix, axis=1)
        intersection = np.diag(confusion_matrix)

        IOU = intersection / (detection_sum + ground_truth_sum - intersection)
        mIOU = np.nanmean(IOU)
        return mIOU

    def return_mAP(self):
        confusion_matrix = self.return_matrix()
        intersection = np.diag(confusion_matrix)
        detection_sum = np.sum(confusion_matrix, axis=0)
        precisions = intersection / detection_sum
        return np.nanmean(precisions)


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
    import torch
    import json
    from tensorboardX import SummaryWriter
    from model.fcos import FCOSDetector
    from dataset.VOC_dataset import VOCDataset
    
    val_dataset = VOCDataset(root_dir='data/VOCdevkit/VOC2007', resize_size=[800, 1333],
                               split='val', use_difficult=False, is_train=False, augment=None)
    print("INFO===>eval dataset has %d imgs"%len(val_dataset))
    val_data_loader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,collate_fn=val_dataset.collate_fn)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model=FCOSDetector(mode="inference")
    model.to(device)

    acc_lst = []
    miou_lst = []
    mAP_lst = []

    writer = SummaryWriter("logs/fcos_val")
    for epoch in range(25):
        # load train weights
        weights_path = f"checkpoint/fcos-{epoch+1}.pth"
        model.load_state_dict({k.replace('module.',''):
                               v for k,v in torch.load(f"checkpoint/fcos-{epoch+1}.pth",map_location=torch.device('cpu')).items()})

        conf_mat = ConfusionMatrix(num_classes=20, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5)
        model.eval()
        
        num = 0
        for i, [images, boxes, classes] in enumerate(val_data_loader):
            images = images.to(device)
            outputs = model(images)

            de = torch.cat((outputs[2][0],outputs[0][0].unsqueeze(1), outputs[1][0].unsqueeze(1)), 1).detach().cpu().numpy()
            lab = torch.cat((classes[0].unsqueeze(1), boxes[0]), 1).cpu().numpy()
            conf_mat.process_batch(de, lab)
            num += 1
            print(f'{epoch}/{num}',end='\r')

        mat = conf_mat.return_matrix()
        acc = conf_mat.return_acc()
        miou = conf_mat.return_mIoU()
        map = conf_mat.return_mAP()

        acc_lst.append(acc)
        miou_lst.append(miou)
        mAP_lst.append(map)

        writer.add_scalar('Acc', acc, epoch + 1)
        writer.add_scalar('mIoU', miou, epoch + 1)
        writer.add_scalar('mAP', map, epoch + 1)

    with open('pascal_voc_classes.json', 'r') as f:
        class_dict = json.load(f)

    plot_confusion_matrix(mat, names=list(class_dict.keys()))
    plt.savefig('img_out/fcos_confusion_matrix.jpg')

    val_criteria = pd.DataFrame({'mIoU': miou_lst, 'acc': acc_lst, 'mAP': mAP_lst})
    val_criteria.to_csv('fcos.csv')