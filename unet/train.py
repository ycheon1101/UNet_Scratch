from dataset import train_dataloader, val_dataloader
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# path
images_path = './images'
images_train_path = './images/train'
images_val_path = './images/val'
model_path = './saved_model.pth'


# param
device = 'cuda:1'
num_epoch = 30
lr = 1e-4
criterion = nn.CrossEntropyLoss(ignore_index=255)

# instanciation model
model = UNet().to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# convert label to color
def convert_label_to_color_img(predicted_tensor):
    color_dict = {
        0: (128, 64, 128),
        1: (244, 35, 232),
        2: ( 70, 70, 70),
        3: (102, 102, 156),
        4: (190, 153, 153),
        5: (153, 153, 153),
        6: (250, 170, 30),
        7: (220, 220, 0),
        8: (107, 142, 35),
        9: (152, 251, 152),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (255,  0,  0),
        13: (0, 0, 142),
        14: (0, 0, 70),
        15: (0, 60, 100),
        16: (0, 80, 100),
        17: (0, 0, 230),
        18: (119, 11, 32)
    }

    # predicted.shape = N, C, H, W
    # print(predicted_tensor.shape)
    
    # print(predicted_label.shape)
    # return
    H, W = predicted_tensor.shape
    # predicted_tensor = predicted_tensor.squeeze(0)

    # print(predicted_tensor.shape)
    color_image = np.zeros((H, W, 3), dtype=np.uint8)

    for class_id, color in color_dict.items():
        mask = (predicted_tensor == class_id)
        color_image[mask] = color

    # color_image = Image.fromarray(color_image)
    return color_image

def save_img(predicted_tensor, img_name, epoch, path):
    predicted_tensor = torch.argmax(predicted_tensor, dim=1)     # [1, 388, 388]
    predicted_tensor = predicted_tensor.squeeze(0)
    predicted_tensor = predicted_tensor.cpu().detach().numpy()
    generated_img = convert_label_to_color_img(predicted_tensor)

    plt.imsave(path + f'/{img_name}/generated_{epoch}.png', generated_img)

# calc IoU
def calculate_iou(pred, target, n_classes):
    ious = []
    pred = np.asarray(pred).copy()
    target = np.asarray(target).copy()
    
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / max(union, 1))
    
    return ious

# calc mIoU
def calculate_miou(ious):
    ious = np.array(ious)
    return np.nanmean(ious)

# save model
def save_model(model, save_path):
    torch.save(model, save_path)

# load model
def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))

# train
def train(dataloader):
    for epoch in range(num_epoch):
        for idx, (img_name, src_img, gt_img) in enumerate(dataloader):
            # print(idx, src_img, gt_img)
            src_img = src_img.to(device)
            gt_img = gt_img.to(device)

            predicted_seg = model(src_img)      # [1, 19, 388, 388]
            
            loss = criterion(predicted_seg, gt_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epoch}, Loss: {loss.item()}")
            if epoch % 5 == 0:
                save_img(predicted_seg, img_name[0], epoch, images_train_path)
            # print(img_name)
    save_model(model.state_dict(), model_path)

# test
def test(dataloader):
    model = UNet().to(device) 
    load_model(model, model_path)
    model.eval()
    
    total_loss = 0
    iou_list = []
    num_class = 19

    with torch.no_grad():
        for img_name, src_img, gt_img in dataloader:
            src_img = src_img.to(device)
            gt_img = gt_img.to(device)

            predicted_seg = model(src_img) 
            loss = criterion(predicted_seg, gt_img)
            total_loss += loss.item()

            _, predicted = torch.max(predicted_seg, 1)
            predicted_np = predicted.cpu().numpy()
            gt_np = gt_img.cpu().numpy()

            # calc IoU
            for cls in range(num_class):
                ious = calculate_iou(predicted_np == cls, gt_np == cls, num_class)
                iou_list.append(ious)

            save_img(predicted_seg, img_name[0], 0, images_val_path)

    miou = calculate_miou([np.nanmean(iou) for iou in zip(*iou_list)])
    
    avg_loss = total_loss / len(dataloader.dataset)
    print(f'Avg Loss: {avg_loss:.4f}, Mean IoU: {miou:.4f}')
            

if __name__ == '__main__':
    train(train_dataloader)
    test(val_dataloader)
