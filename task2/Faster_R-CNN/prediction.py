import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision import transforms
from networks import FasterRCNN, FCOS, AnchorsGenerator
from backbone import MobileNetV2
from draw_box_utils import draw_objs


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


def create_model_fcos(num_classes):
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((8,), (16,), (32,), (64,), (128,)),
                                    aspect_ratios=((1.0,),(1.0,),(1.0,),(1.0,),(1.0,)))

    # put the pieces together inside a FCOS model
    model = FCOS(backbone,
             num_classes = num_classes,
             anchor_generator = anchor_generator)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def plot_predict(image_path, model_name = 'faster_rcnn', out_path = './img_out'):
    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if model_name == 'faster_rcnn':
        # create model
        model = create_model_faster_rcnn(num_classes=21)

        # load train weights
        weights_path = "./save_weights/faster-rcnn-24.pth"
        assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)

    elif model_name == 'fcos':
        # create model
        model = create_model_fcos(num_classes=21)

        # load train weights
        weights_path = "./save_weights/fcos-24.pth"
        assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)

    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = Image.open(image_path)

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.2,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=50)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save(Path(out_path)/f"{model_name}_{Path(image_path).name}")


if __name__ == '__main__':
    for img in Path('./test_images').glob('*jpg'):
        plot_predict(img, 'fcos')
