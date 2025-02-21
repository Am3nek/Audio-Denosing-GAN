import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dual_conv = nn.Sequential(
            nn.Conv1d(in_channels,mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.dual_conv(x)

class Downscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=4)
    
    def forward(self, x):
        return self.maxpool(x)
    

class UpScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upConv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4)
    
    def forward(self, x):
        return self.upConv(x)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_1 = nn.Transformer(
            nhead=6,
            num_encoder_layers=3,
            num_decoder_layers=3
        )

    def forward(self, x):
        return self.transformer_1(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Downsample
        self.conv_1 = ConvBlock(1,32,32)
        self.conv_2 = ConvBlock(32,64,64)
        self.conv_3 = ConvBlock(64,128,128)
        self.conv_4 = ConvBlock(128,256,256)

        self.deconv_1 = ConvBlock(256,128,128)
        self.deconv_2 = ConvBlock(128,64,64)
        self.deconv_3 = ConvBlock(64,32,32)
        self.deconv_4 = ConvBlock(32,1,16)

        #Downscale
        self.down_1 = Downscale()

        #Bridge
        self.bridge = ConvBlock(256,256,256)

        #Upscale
        self.up_scale1 = UpScale(256,256)
        self.up_scale2 = UpScale(128,128)
        self.up_scale3 = UpScale(64,64)
        self.up_scale4 = UpScale(32,32)

        #Transformer
        self.down_trans_1 = Transformer()
        self.down_trans_2 = Transformer()
        self.down_trans_3 = Transformer()
        self.down_trans_4 = Transformer()

        self.up_trans_1 = Transformer()
        self.up_trans_2 = Transformer()
        self.up_trans_3 = Transformer()
        self.up_trans_4 = Transformer()

        #Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Downsample path
        conv1 = self.conv_1(x)
        trans1 = self.down_trans_1(conv1)
        pool1 = self.down_1(trans1)
        
        conv2 = self.conv_2(pool1)
        trans2 = self.down_trans_2(conv2)
        pool2 = self.down_1(trans2)
        
        conv3 = self.conv_3(pool2)
        trans3 = self.down_trans_3(conv3)
        pool3 = self.down_1(trans3)
        
        conv4 = self.conv_4(pool3)
        trans4 = self.down_trans_4(conv4)
        pool4 = self.down_1(trans4)
        
        bridge = self.bridge(pool4)
        
        up1 = self.up_scale1(bridge)
        up1_trans = self.up_trans_1(up1)
        merge1 = torch.cat([up1_trans, trans4], dim=1)
        deconv1 = self.deconv_1(merge1)
        
        up2 = self.up_scale2(deconv1)
        up2_trans = self.up_trans_2(up2)
        merge2 = torch.cat([up2_trans, trans3], dim=1)
        deconv2 = self.deconv_2(merge2)
        
        up3 = self.up_scale3(deconv2)
        up3_trans = self.up_trans_3(up3)
        merge3 = torch.cat([up3_trans, trans2], dim=1)
        deconv3 = self.deconv_3(merge3)
        
        up4 = self.up_scale4(deconv3)
        up4_trans = self.up_trans_4(up4)
        merge4 = torch.cat([up4_trans, trans1], dim=1)
        deconv4 = self.deconv_4(merge4)
        
        # Final adaptive pooling
        output = self.adaptive_pool(deconv4)
        
        return output