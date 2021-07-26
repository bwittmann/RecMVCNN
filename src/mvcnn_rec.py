"""Deep learning models."""

import torch
import torch.nn as nn
import torchvision.models as models


class ReconstructionMVCNN(nn.Module):
    """Main module of RecMVCNN.

    This architecture is able to classify and reconstruct a 3D shape via multi-view images.
    The model is inspired by the Pix2Vox-F and MVCNN approach.
    """
    def __init__(self, num_classes, backbone_type, no_reconstruction, use_fusion, cat_cls_res, dropout_prob):
        """Initialization method.

        Args:
            num_classes: int indicating the number of individual classes in the dataset
            backbone_type: str describing the type of the 2D CNN backbone
            no_reconstruction: bool indicating to only classify the 3D content
            use_fusion: bool indicating the use of a fusion module to refine the 3D reconstructions
            cat_cls_res: make use of the classification output for the reconstruction task
            dropout_prob: float describing the dropout probability in the classification network
        """
        super().__init__()
        self.num_classes = num_classes
        self.no_reconstuction = no_reconstruction
        self.use_fusion = use_fusion
        self.cat_cls_res = cat_cls_res

        # Backbone to extract 2D features
        self.features = Backbone(backbone_type)
        in_features = self.features.in_features
        in_channels = self.features.in_channels

        # Decoder to decode reconstructions from 2D featues of each image
        self.decoder = Decoder(in_channels)

        # Classifier for the classification task from 2D image features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=dropout_prob, inplace=False),
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=dropout_prob, inplace=False),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

        # Conv layer for incorporating classification results in reconstruction feature map
        self.fuse_cls_res = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64+self.num_classes, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward pass of RecMVCNN.

        Args:
            x: a torch.tensor consisting of RBG images

        Returns:
            two torch.tensors representing the classification and 
            reconstruction predicions
        """
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
        if self.no_reconstuction:
            return cls_ret, None

        # Concatenate classification results to feature list to improve reconstruction
        if self.cat_cls_res:
            #pred_labels = torch.argmax(cls_ret, dim=1)[:, None, None, None]
            pred_labels = cls_ret[:, :, None, None]
            pred_labels = pred_labels.expand(-1, -1, 5, 5)

            for idx, feature in enumerate(feature_list):
                feature_list[idx] = self.fuse_cls_res(torch.cat((feature, pred_labels), dim=1))

        generated_volume_list, raw_decoded_feature_list = self.decoder(feature_list)

        if self.use_fusion:
            # TODO: implement
            pass
        else:
            rec_ret = torch.mean(torch.stack(generated_volume_list, dim=0), dim=0)
            return cls_ret, rec_ret


class FusionModule(nn.Module):
    """A fusion module to refine the 3D reconstructions."""
    def __init__(self):
        super().__init__()
        # TODO: implement
        pass


class Backbone(nn.Module):
    """Pre-trained CNN backbone for the 2D feature extraction."""
    def __init__(self, backbone_type):
        """Initialization method.

        Args: Type of the backbone for feature extraction
        """
        super().__init__()
        # Backbone for the 2D feature extraction
        if backbone_type == 'vgg16_1x1conv':
            vgg = models.vgg16(pretrained=True)
            self.features = nn.Sequential(
                vgg.features,
                nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.in_features = 64*4*4
            self.in_channels = 128
        elif backbone_type == 'resnet18_1x1conv': # num params: 11.2M, out dim: [B, 512, 5, 5]
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(
                *list(resnet.children())[:-2],
                nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.in_features = 64*5*5
            self.in_channels = 200
        elif backbone_type == 'mobilenetv3l_1x1conv': # num params: 3.0M, out dim: [B, 960, 5, 5]
            mobnet = models.mobilenet_v3_large(pretrained=True)
            self.features = nn.Sequential(
                *list(mobnet.children())[:-2],
                nn.Conv2d(in_channels=960, out_channels=64, kernel_size=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.in_features = 64*5*5
            self.in_channels = 200
        elif backbone_type == 'mobilenetv3s_1x1conv': # num params: 930k, out dim; [B, 576, 5, 5]
            mobnet = models.mobilenet_v3_small(pretrained=True)
            self.features = nn.Sequential(
                *list(mobnet.children())[:-2],
                nn.Conv2d(in_channels=576, out_channels=64, kernel_size=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.in_features = 64*5*5
            self.in_channels = 200
        elif backbone_type == 'resnet18_stdconv':
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(
                *list(resnet.children())[:-4],
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
            self.in_features = 64*4*4
            self.in_channels = 128
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward pass of the backbone.

        Args:
            x: a RBG image

        Returns:
            Fetures extracted from the RGB image
        """
        return self.features(x)


class Decoder(nn.Module):
    """The reconstruction branch.
    
    Takes as input a list of features from 2D multi-view images and estimates a 3D 
    reconstruction in the form of a occupancy voxel grid.
    """
    def __init__(self, in_channels):
        """Initialization method.
        
        Params:
            in_channels: channels of the feature representations
        """
        super().__init__()
        # Decoder for the reconstruction of 3D features
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
            nn.Sigmoid() # to deal with problem as a binary classification task
        )

    def forward(self, feature_list):
        """Forward pass of the reconstruction decoder.

        Args:
            feature_list: a list containing the features of the multi-view images

        Returns:
            the 32x32x32 predicted reconstruction of the 3D shape
        """
        batch_size = feature_list[0].shape[0]

        raw_decoded_features_list = []
        generated_volume_list = []

        # Decode view_features into decoded features and generated volumes for reconstruction
        for view_features in feature_list:
            view_features = view_features.view(batch_size, -1, 2, 2, 2) # [B, C, 2, 2, 2]
            decoded_features = self.decoder_features(view_features) # [B, 8, 32, 32, 32]
            raw_decoded_features = decoded_features

            generated_volume = self.decoder_volume(decoded_features) # [B, 1, 32, 32, 32]

            raw_decoded_features = torch.cat((raw_decoded_features, generated_volume), dim=1) # [B, 9, 32, 32, 32]

            generated_volume_list.append(generated_volume.squeeze())
            raw_decoded_features_list.append(raw_decoded_features)

        return generated_volume_list, raw_decoded_features_list
