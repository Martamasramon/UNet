import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=7):
        super(InitialBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)

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
        
        self.same_channels = input_size == output_size

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size)
        )
        
        self.projection = nn.Conv2d(input_size, output_size, kernel_size=1) if not self.same_channels else nn.Identity()


    def forward(self, x):
        residual = self.projection(x)
        return residual + self.layers(x)


class RefineBlock(nn.Module):
    def __init__(self, input_size, output_size, upscale_factor=2):
        super(RefineBlock, self).__init__()
        
        channels_after_shuffle = input_size // (upscale_factor ** 2)
        
        self.pre_shuffle = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.Conv2d(input_size, output_size * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor)
        )
        
        # Post-shuffle refinement to remove checkerboard artifacts
        self.post_shuffle = nn.Sequential(
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size)
        )
        
        # Residual connection 
        self.use_residual = True

    def forward(self, x):
        x = self.pre_shuffle(x)
        
        if self.use_residual:
            x = self.post_shuffle(x) + x  # Residual connection
        else:
            x = self.post_shuffle(x)
            
        return F.relu(x)

class FinalBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(FinalBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, output_size, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.layers(x)

# --- Encoder ---

class RUNetEncoder(nn.Module):
    """
    Modified RUNet encoder for embedding extraction.
    Removes skip connections and focuses on bottleneck representation.
    """
    def __init__(self, drop_first=0.1, drop_last=0.4,embedding_dim=512):
        super().__init__()
        drop_2 = drop_first
        drop_3 = drop_first + (drop_last - drop_first) / 3
        drop_4 = drop_first + 2 * (drop_last - drop_first) / 3
        drop_5 = drop_last

        self.pool = nn.MaxPool2d(2)

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

        self.representation = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(0.25),  # Moderate dropout at bottleneck
            nn.Conv2d(1024, embedding_dim, 3, padding=1),  # Direct to embedding_dim
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU()
        )
                
        self.embedding_projection = nn.Sequential(
            nn.AvgPool2d(kernel_size=8), # nn.AdaptiveAvgPool2d((1, 1))
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        """Returns spatial feature map before global pooling"""
        x = self.block1(x)
        x = self.block2(self.pool(x))
        x = self.block3(self.pool(x))
        x = self.block4(self.pool(x))
        x = self.block5(self.pool(x))
        x = self.representation(x)
        out = self.embedding_projection(x)  
        return out


# --- Decoder ---

class RUNetDecoder(nn.Module):
    def __init__(self, embedding_dim=512, initial_size=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.initial_size = initial_size
        
        # Project global embedding back to spatial representation. Assuming input images are 256x256, we need to go from embedding to 8x8x512 feature maps
        self.embedding_to_spatial = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, 512 * initial_size * initial_size),
            nn.ReLU()
        )
                
        # Initial processing of the reconstructed spatial features
        self.initial_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.refine4 = RefineBlock(embedding_dim, 256)     # Start from the representation features (512 channels)
        self.refine3 = RefineBlock(256, 128)  
        self.refine2 = RefineBlock(128, 64)  
        self.refine1 = RefineBlock(64, 32)  
        
        self.final = FinalBlock(32, 16, 1)    

    def forward(self, embed):
        x = embedding_to_spatial(embed)
        x = x.view(embed.size(0), 512, self.initial_size, self.initial_size)  #[B, 512, 8, 8]
        
        # Process the spatial features
        x = self.initial_conv(x)  # [B, 512, 8, 8]
        x = self.refine4(x)
        x = self.refine3(x)
        x = self.refine2(x)
        x = self.refine1(x)
        out = self.final(x)
        return out

# --- Full Autoencoder Wrapper ---

class RUNet(nn.Module):
    def __init__(self, drop_first, drop_last, embedding_dim=512):
        super().__init__()
        self.encoder = RUNetEncoder(drop_first=drop_first, drop_last=drop_last, embedding_dim=embedding_dim)
        self.decoder = RUNetDecoder(embedding_dim=embedding_dim)

    def forward(self, x):
        embedding       = self.encoder(x)
        reconstruction  = self.decoder(embedding)
        
        return reconstruction

    def get_embedding(self, x):
        embedding = self.encoder(x)
        return embedding
    