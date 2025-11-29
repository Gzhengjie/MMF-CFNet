import torch
from torch.utils.data import DataLoader
import sys
import os
import pandas as pd
import numpy as np
import time
import yaml
from models.mmf_cfnet.model import MMF_CFNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)

with open("configs/mmfnet.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths_cfg = cfg["paths"]
train_cfg = cfg["training"]
model_cfg = cfg["model"]

model_path = paths_cfg["model_dir"]
log_path = os.path.join(model_path, "Logs")
dataset_dir = paths_cfg["dataset_dir"]
save_model_path = os.path.join(model_path, paths_cfg["model_name"])

# 自动创建目录
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

# 生成日志文件名
train_csv_file_name = os.path.join(log_path, f'train-{time.strftime("%Y%m%d-%H%M%S")}.csv')
valid_csv_file_name = os.path.join(log_path, f'valid-{time.strftime("%Y%m%d-%H%M%S")}.csv')

model = MMF_CFNet(
    encoder_name=model_cfg["encoder_name"],
    encoder_name1=model_cfg["encoder_name1"],
    encoder_name2=model_cfg["encoder_name2"],
    encoder_name3=model_cfg["encoder_name3"],
    encoder_depth=model_cfg["encoder_depth"],
    in_channels=model_cfg["in_channels"],
    in_channels1=model_cfg["in_channels1"],
    in_channels2=model_cfg["in_channels2"],
    in_channels3=model_cfg["in_channels3"],
    encoder_weights=None,
    fusion_form=model_cfg["fusion_form"],
    siam_encoder=model_cfg["siam_encoder"],
    classes=model_cfg["n_classes"],
)

# -------------------------------------------------------------
# 🔥 构建数据集
# -------------------------------------------------------------
size = train_cfg["input_size"]
batch_size_train = train_cfg["batch_size_train"]
batch_size_val = train_cfg["batch_size_val"]

train_dataset = LEVIR_SEG_Dataset(
    os.path.join(dataset_dir, "train_image"),
    img_suffix='.tif',
    ann_dir=os.path.join(dataset_dir, "train_label"),
    debug=False,
    size=size,
)

valid_dataset = LEVIR_SEG_Dataset(
    os.path.join(dataset_dir, "val_image"),
    img_suffix='.tif',
    ann_dir=os.path.join(dataset_dir, "val_label"),
    debug=False,
    test_mode=True,
    size=size,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False, num_workers=0, drop_last=True)

# -------------------------------------------------------------
loss = utils.losses.CrossEntropyLoss()

metrics = [
    utils.metrics.Fscore(activation='argmax2d'),
    utils.metrics.Precision(activation='argmax2d'),
    utils.metrics.Recall(activation='argmax2d'),
    utils.metrics.Accuracy(activation='argmax2d'),
    utils.metrics.IoU(activation='argmax2d', n_classes=model_cfg["n_classes"]),
    utils.metrics.FWIou(activation='argmax2d', n_classes=model_cfg["n_classes"]),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=train_cfg["lr"]),
])

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

train_epoch = utils.train_seg.TrainEpoch(model, loss, metrics, optimizer, DEVICE, True)
valid_epoch = utils.train_seg.ValidEpoch(model, loss, metrics, DEVICE, True)

# -------------------------------------------------------------
max_score = 0
MAX_EPOCH = train_cfg["epochs"]

# 初始化 CSV
df_train = pd.DataFrame(columns=['cross_entropy_loss','fscore','precision','recall','accuracy','iou_score','FWIou_score'])
df_valid = pd.DataFrame(columns=['cross_entropy_loss','fscore','precision','recall','accuracy','iou_score','FWIou_score'])
df_train.to_csv(train_csv_file_name, index=False)
df_valid.to_csv(valid_csv_file_name, index=False)

for epoch in range(MAX_EPOCH):
    print("\nEpoch:", epoch)

    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # 写入 CSV
    pd.DataFrame([list(train_logs.values())]).to_csv(train_csv_file_name, mode='a', header=False, index=False)
    pd.DataFrame([list(valid_logs.values())]).to_csv(valid_csv_file_name, mode='a', header=False, index=False)

    scheduler_steplr.step()

    if valid_logs['fscore'] > max_score:
        max_score = valid_logs['fscore']
        print("fscore:", max_score)
        torch.save(model, save_model_path)
        print("Model saved!")

