import torch
import torch.nn as nn
from torch.nn.modules import padding
import torchvision.models as models


class MVCNNRec(nn.Module):

    def __init__(self, num_classes, backbone):
        """
        Inspired by:
        - https://github.com/RBirkeland/MVCNN-PyTorch
        - https://github.com/hzxie/Pix2Vox/tree/Pix2Vox-F
        """
        super().__init__()
        self.num_classes = num_classes

        # Backbone for the 2D feature extraction
        if backbone == 'vgg16': # num params: 14.7M, out dim: [B, 512, 4, 4]
            vgg = models.vgg16(pretrained=True)
            self.features = vgg.features
            in_features = 512*4*4
            in_channels = 1024
        elif backbone == 'resnet18': # num params: 11.2M, out dim: [B, 512, 5, 5]
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            in_features = 512*5*5
            in_channels = 1600
        elif backbone == 'mobilenetv3l': # num params: 3.0M, out dim: [B, 960, 5, 5]
            mobnet = models.mobilenet_v3_large(pretrained=True)
            self.features = nn.Sequential(*list(mobnet.children())[:-2])
            # TODO: too big -> reduce if possible or use pooling? do we loose spacial infos?
            in_features = 960*5*5
            in_channels = 3000
        elif backbone == 'mobilenetv3s': # num params: 930k, out dim; [B, 576, 5, 5]
            mobnet = models.mobilenet_v3_small(pretrained=True)
            self.features = nn.Sequential(*list(mobnet.children())[:-2])
            in_features = 576*5*5
            in_channels = 1800
        elif backbone == 'test':
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(
                *list(resnet.children())[:-4],
                # Additional layers to reduce channels and spacial dim simulataneously ([64,4,4] vs [64,7,7])
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            in_features = 64*4*4
            in_channels = 128
        else:
            raise NotImplementedError

        # Classifier for the classification task from 2D images
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            # TODO: think about cutting this layer
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features= 4096, out_features=num_classes)
        )

        # Decoder for the reconstruction of 3D features
        # TODO: think of adding bias, num channels too much (vanilla version: 392 channels) -> network that reduces channels and spacial dim
        self.decoder_features = nn.Sequential(
            # Layer 1: out [B, 128, 4, 4, 4]
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            # Layer 2: out [B, 64, 8, 8, 8]
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Layer 3: out [B, 32, 16, 16, 16]
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Layer 4: out [B, 32, 32, 32, 32]
            nn.ConvTranspose3d(in_channels=32, out_channels=8, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # Decoder for the reconstruction of 3D volumes from 3D features (1x1 conv)
        self.decoder_volume = nn.Sequential(
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Use shared backbone to extract features of input images
        x = x.transpose(0, 1) # [V, B, 3, H, W] rgb images

        feature_list = []
        for view in x:
            view_features = self.features(view) # see backbones for shape
            feature_list.append(view_features)

        # View pooling for classification results
        max_features = feature_list[0].view(view_features.shape[0], -1)
        for view_features in feature_list[0:]:
            view_features = view_features.view(view_features.shape[0], -1)
            max_features = torch.max(max_features, view_features) # [B, in_features]

        # Get classificaton return
        cls_ret = self.classifier(max_features) # [B, num_classes]

        # Decode view_features into decoded features and generated volumes
        raw_decoded_features_list = []
        generated_volume_list = []
        # Decoding of features for reconstruction
        for view_features in feature_list:
            view_features = view_features.view(batch_size, -1, 2, 2, 2) # [B, C, 2, 2, 2]
            decoded_features = self.decoder_features(view_features) # [B, 8, 32, 32, 32]
            raw_decoded_features = decoded_features

            generated_volume = self.decoder_volume(decoded_features) # [B, 1, 32, 32, 32]

            raw_decoded_features = torch.cat((raw_decoded_features, generated_volume), dim=1) # [B, 9, 32, 32, 32]

            generated_volume_list.append(generated_volume.squeeze())
            raw_decoded_features_list.append(raw_decoded_features)

        return cls_ret #, generated_volume_list, raw_decoded_features_list