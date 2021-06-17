import torch
import torch.nn as nn
import torchvision.models as models


class MVCNN(nn.Module):

    def __init__(self, num_classes=40, num_views=12):
        super(MVCNN, self).__init__()
        self.num_classes = num_classes
        # I do not know whether this parameter might be needed for reshaping the images
        # Probably this value can be inferred from the shape of the input x!
        self.num_views = num_views

        vgg = models.vgg16(pretrained=True)
        self.vgg_features = vgg.features
        self.vgg_classifier = vgg.classifier
        self.vgg_classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        print("shape of x", x.shape)
        y = self.vgg_features(x)
        print("shape after vgg_features", y.shape)
        # I am unsure if this is the correct way to reshape the output of the classifier!
        y = y.view((int(x.shape[0]/self.num_views),
                   self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))
        y = torch.max(y,1)[0].view(y.shape[0],-1)
        print("shape before classification", y.shape)
        y = self.vgg_classifier(y)
        print("shape after classification", y.shape)
        return y
