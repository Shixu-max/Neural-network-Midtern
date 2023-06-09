import math
import sys
import time

import torch

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .confusion_matrix import ConfusionMatrix
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    mlosses = {"classification": torch.zeros(1).to(device),
               "bbox_regression": torch.zeros(1).to(device),
               "bbox_ctrness": torch.zeros(1).to(device)}

    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict, detections = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        for loss in mlosses.keys():
            mlosses[loss] = (mlosses[loss]*i + loss_dict_reduced[loss]) / (i+1)

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, mlosses, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    mloss = torch.zeros(1).to(device)  # mean losses
    conf_mat = ConfusionMatrix(num_classes=20, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5)  # confusion matrix

    model.eval()
    for i, [image, targets] in enumerate(metric_logger.log_every(data_loader, 100, header)):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        loss_dict, outputs = model(image, targets)

        # 记录验证损失
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        mloss += loss_value  # update sum losses
        model_time = time.time() - model_time

        # mIoU and acc
        de = torch.cat((outputs[0]['boxes'],
                        outputs[0]['scores'].unsqueeze(1), outputs[0]['labels'].unsqueeze(1)), 1).detach().cpu().numpy()
        lab = torch.cat((targets[0]['labels'].unsqueeze(1), targets[0]['boxes']), 1).detach().cpu().numpy()
        conf_mat.process_batch(de, lab)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list
    mIoU = conf_mat.return_mIoU()
    acc = conf_mat.return_acc()

    return mloss/(i+1), coco_info, mIoU, acc


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
