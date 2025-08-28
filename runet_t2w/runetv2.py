import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InitialBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=7):
        super(InitialBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(ResBlock, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            SEBlock(output_size)
        )
        
        self.relu = nn.ReLU()

        self.match_dim = None
        if input_size != output_size:
            self.match_dim = nn.Conv2d(input_size, output_size, kernel_size=1)

    def forward(self, x):
        identity = self.match_dim(x) if self.match_dim else x
        x        = self.layers(x)
        return self.relu(x + identity)


class RefineBlock(nn.Module):
    def __init__(self, input_size, output_size, upscale_mode='nearest'):
        super(RefineBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode=upscale_mode)

        self.res_conv = nn.Conv2d(input_size, output_size, kernel_size=1)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_size)
        )

    def forward(self, x):
        residual = self.upsample(x)
        residual = self.res_conv(residual)

        out = self.conv(x)
        out = self.upsample(out)

        return out + residual


class FinalBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(FinalBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, output_size, kernel_size=1),
            nn.Tanh()  # assuming output image is normalized to [-1, 1]
        )

    def forward(self, x):
        return self.layers(x)


class RUNet(nn.Module):
    def __init__(self, drop_first, drop_last, img_size):
        super(RUNet, self).__init__()
        drop_2 = drop_first
        drop_3 = drop_first + (drop_last - drop_first) / 3
        drop_4 = drop_first + 2 * (drop_last - drop_first) / 3
        drop_5 = drop_last

        if drop_first == 0.0:
            drop_2 = drop_3 = drop_4 = 0.0
            
        self.img_size = img_size

        self.block1 = InitialBlock(1, 64)

        self.block2 = nn.Sequential(
            ResBlock(64, 64, drop_2),
            ResBlock(64, 64, drop_2),
            ResBlock(64, 64, drop_2),
            ResBlock(64, 128, drop_2)
        )

        self.block3 = nn.Sequential(
            ResBlock(128, 128, drop_3),
            ResBlock(128, 128, drop_3),
            ResBlock(128, 128, drop_3),
            ResBlock(128, 256, drop_3)
        )

        self.block4 = nn.Sequential(
            ResBlock(256, 256, drop_4),
            ResBlock(256, 256, drop_4),
            ResBlock(256, 256, drop_4),
            ResBlock(256, 256, drop_4),
            ResBlock(256, 256, drop_4),
            ResBlock(256, 512, drop_4)
        )

        self.block5 = nn.Sequential(
            ResBlock(512, 512, drop_5),
            ResBlock(512, 512, drop_5),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.representation_transform = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder starts from the transformed representation only (no skip connections)
        self.refine4 = RefineBlock(512, 512)
        self.refine3 = RefineBlock(512, 384)
        self.refine2 = RefineBlock(384, 256)
        self.refine1 = RefineBlock(256, 96)

        self.final = FinalBlock(96, 99, 1)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(self.max_pool(x1))
        x3 = self.block3(self.max_pool(x2))
        x4 = self.block4(self.max_pool(x3))
        
        if self.img_size == 128:
            x5    = self.block5(self.max_pool(x4))
            embed = self.representation_transform(x5)
            out4  = self.refine4(embed)
            out3  = self.refine3(out4)
        else:
            embed  = self.representation_transform(x4)  
            out3 = self.refine3(embed)      
        
        out2 = self.refine2(out3)
        out1 = self.refine1(out2)

        output = self.final(out1)

        return output, embed

    def get_embedding(self, x):
        
        x1 = self.block1(x)
        x2 = self.block2(self.max_pool(x1))
        x3 = self.block3(self.max_pool(x2))
        x4 = self.block4(self.max_pool(x3))
        
        if self.img_size == 128:
            x5    = self.block5(self.max_pool(x4))
            embed = self.representation_transform(x5)
        else:
            embed = self.representation_transform(x4)

        return embed
    
    def get_all_embeddings(self, x):
        
        x1 = self.block1(x)
        x2 = self.block2(self.max_pool(x1))
        x3 = self.block3(self.max_pool(x2))
        x4 = self.block4(self.max_pool(x3))
        
        if self.img_size == 128:
            return x1, x2, x3, x4
        else:
            return x, x1, x2, x3
    
    #### 
    # torch.Size([1, 1, 128, 128]) 
    # torch.Size([1, 64, 128, 128]) 
    # torch.Size([1, 128, 64, 64]) 
    # torch.Size([1, 256, 32, 32]) 
    # torch.Size([1, 512, 16, 16]) 
    # torch.Size([1, 512, 8, 8]) 
    # torch.Size([1, 512, 4, 4])