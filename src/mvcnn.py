import torch
import torch.nn as nn
import torchvision.models as models


class MVCNN(nn.Module):

    def __init__(self, num_classes, backbone):
        """
        Inspired by https://github.com/RBirkeland/MVCNN-PyTorch.
        """
        super(MVCNN, self).__init__()
        self.num_classes = num_classes

        # TODO add more options
        if backbone == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            self.features = vgg.features
            # Implement own classifier as we have different image size.
            self.classifier = nn.Sequential(
                nn.Linear(in_features=512*4*4, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features= 4096, out_features=num_classes)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        # Use shared backbone to extract features of input images
        x = x.transpose(0, 1)

        feature_list = []
        for view in x:
            view_features = self.features(view)
            view_features = view_features.view(view_features.shape[0], -1)
            feature_list.append(view_features)

        # View pooling
        max_features = feature_list[0]
        for view_features in feature_list[0:]:
            max_features = torch.max(max_features, view_features)

        ret = self.classifier(max_features)
        return ret