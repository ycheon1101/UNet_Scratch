import torch
import torch.nn as nn
import pandas as pd
# from dataset import train_dataloader
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

# repeated conv (kernel = 3 * 3, stride = 1, padding = none, activation = Relu)
def repeat_conv(in_channel, out_channel):
    repeat_conv_layers = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1),
        nn.ReLU(inplace=True)
    )
    return repeat_conv_layers

# Unet class
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # apply initilize weight
        # self.apply(initialize_weights)
        self.in_channel = 3
        self.num_class = 19

        # max pool layer: kernel = 2 * 2, stride = 2
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        self.encoder_1 = repeat_conv(in_channel=self.in_channel, out_channel=64)
        self.encoder_2 = repeat_conv(in_channel=64, out_channel=128)
        self.encoder_3 = repeat_conv(in_channel=128, out_channel=256)
        self.encoder_4 = repeat_conv(in_channel=256, out_channel=512)
        self.encoder_5 = repeat_conv(in_channel=512, out_channel=1024)

        # drop out at the end of contracting path
        self.dropout = nn.Dropout(0.3)

        # up_sampling (kernel size = 2, stride = 2)
        self.upsampling_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.upsampling_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upsampling_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upsampling_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        # decoder
        self.decoder_4 = repeat_conv(in_channel=1024, out_channel=512)
        self.decoder_3 = repeat_conv(in_channel=512, out_channel=256)
        self.decoder_2 = repeat_conv(in_channel=256, out_channel=128)
        self.decoder_1 = repeat_conv(in_channel=128, out_channel=64)

        # out
        self.out = nn.Conv2d(in_channels=64, out_channels=self.num_class, kernel_size=1)
    
    # crop image to concat down sampling layer and up sampling layer -> enc tensor should be cropped
    def crop_img(self, enc_tensor, dec_tensor):
        # extract size
        enc_height = enc_tensor.shape[-2]
        enc_width = enc_tensor.shape[-1]
        dec_height = dec_tensor.shape[-2]
        dec_width = dec_tensor.shape[-1]

        # enc_width > dec_width
        diff_size_width = enc_width - dec_width 
        diff_size_height = enc_height - dec_height 
        diff_range_width = diff_size_width // 2
        diff_range_height = diff_size_height // 2
        check_height = enc_height - 2 * diff_range_height
        check_width = enc_width - 2 * diff_range_width
        diff_height = check_height - dec_height
        diff_width = check_width - dec_width

        return enc_tensor[:, :, diff_range_height : enc_height - diff_range_height - diff_height, diff_range_width : enc_width - diff_range_width - diff_width]
        
    # concat with enc_tensor and dec_tensor
    def concat(self, enc_tensor, dec_tensor):
        enc_tensor = self.crop_img(enc_tensor, dec_tensor)
        return torch.cat([enc_tensor, dec_tensor], axis=1)

    def forward(self, x):
        # x.shape = [b, c, h, w] = [1, 3, 1024, 2048]
        # contracting path
        # print(x.shape)
        enc_layer_1 = self.encoder_1(x)                   # [1, 64, 1020, 2044]
        enc_max_pool_1 = self.max_pool(enc_layer_1)       # [1, 64, 510, 1022]

        enc_layer_2 = self.encoder_2(enc_max_pool_1)      # [1 ,128, 506, 1018] 
        enc_max_pool_2 = self.max_pool(enc_layer_2)       # [1, 128, 253, 509]

        enc_layer_3 = self.encoder_3(enc_max_pool_2)      # [1, 256, 249, 505]
        enc_max_pool_3 = self.max_pool(enc_layer_3)       # [1, 256, 124, 252]

        enc_layer_4 = self.encoder_4(enc_max_pool_3)      # [1, 512, 120, 248]
        enc_max_pool_4 = self.max_pool(enc_layer_4)       # [1, 512, 60, 124]

        enc_layer_5 = self.encoder_5(enc_max_pool_4)      # [1, 1024, 56, 120]
        enc_layer_5 = self.dropout(enc_layer_5)           # drop out

        # # expansive path with concat
        dec_layer4 = self.upsampling_1(enc_layer_5)       # [1, 512, 112, 240]
        dec_layer4 = self.concat(enc_layer_4, dec_layer4) # [1, 1024, 112, 240]  concat encoder layer4 and decoder layer 4
        dec_layer4 = self.decoder_4(dec_layer4)           # [1, 512, 108, 236]

        dec_layer3 = self.upsampling_2(dec_layer4)        # [1, 256, 216, 472]
        dec_layer3 = self.concat(enc_layer_3, dec_layer3) # [1, 512, 216, 472]
        dec_layer3 = self.decoder_3(dec_layer3)           # [1, 256, 212, 468]

        dec_layer2 = self.upsampling_3(dec_layer3)        # [1, 128, 424, 936]
        dec_layer2 = self.concat(enc_layer_2, dec_layer2) # [1, 256, 424, 936]
        dec_layer2 = self.decoder_2(dec_layer2)           # [1, 128, 420, 932]

        dec_layer1 = self.upsampling_4(dec_layer2)        # [1, 64, 840, 1864]
        dec_layer1 = self.concat(enc_layer_1, dec_layer1) # [1, 128, 840, 1864]
        dec_layer1 = self.decoder_1(dec_layer1)           # [1, 64, 836, 1860]

        # # seg_tensor
        output = self.out(dec_layer1)                     # [1, 19, 836, 1860]
        

        return output





