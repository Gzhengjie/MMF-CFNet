import os.path as osp
import torch
from tqdm import tqdm as tqdm
import cv2
import sys
import numpy as np


def check_tensor(data, is_label):
    if not is_label:
        return data if data.ndim <= 4 else data.squeeze()
    return data.long() if data.ndim <= 3 else data.squeeze().long()


model = torch.load(r".\MMF-CFNet-main\ResNext50_32×4d\ResNext50_32×4d+MobileNetV4-S_model.pth", map_location='cuda:0')
# eval()框架会自动把BN和DropOut固定住，不会取平均，而且用训练好的值
model.eval()

# valid_dataset = LEVIR_CD_Dataset('GF2-CD/val',
#                                  sub_dir_1='A',
#                                  sub_dir_2='B',
#                                  img_suffix='.tif',
#                                  ann_dir='GF2-CD/val/label',
#                                  debug=False,
#                                  test_mode=True)
valid_dataset = LEVIR_SEG_Dataset(r'.\QTPGSD\test_image',
                                 img_suffix='.tif',
                                 ann_dir=r'.\QTPGSD\test_label',
                                 debug=False,
                                 test_mode=True)

dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
device = 'cuda'
metrics = [
    metrics.Fscore(activation='argmax2d'),
    metrics.Recall(activation='argmax2d'),
    metrics.Dice(activation='argmax2d'),
    metrics.Accuracy(activation='argmax2d'),
    metrics.IoU(activation='argmax2d', n_classes=2),
    metrics.Fscore_each(activation='argmax2d')
]
for metric in metrics:
    metric.to(device)

metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}

with tqdm(dataloader, desc="val", file=sys.stdout) as iterator:
    for (x, y, filename) in iterator:
        x, y = check_tensor(x, False), \
                    check_tensor(y, True)
        # print(x.shape)
        a = y.numpy()[0]
        x, y = x.to(device), y.to(device)
        y_pred = model.forward(x)


        y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
        filename = filename[0].split('.')[0] + ".tif"
        save_dir = r".\QTPGSD\test_output"
        cv2.imwrite(osp.join(save_dir, filename), a)
        print(osp.join(save_dir, filename))

