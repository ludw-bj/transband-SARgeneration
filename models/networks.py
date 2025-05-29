import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self, phase = 'test'):
        super(Vgg16, self).__init__()
        self.phase = phase
        features = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

class Vgg19(nn.Module):
    def __init__(self, contentLayer, styleLayer, phase = 'test'):
        super(Vgg19, self).__init__()
        self.contentLayer = contentLayer
        self.styleLayer = styleLayer
        self.phase = phase
        features = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1).features
        self.to_relu_1 = nn.Sequential() 
        self.to_relu_2 = nn.Sequential() 
        self.to_relu_3 = nn.Sequential()
        self.to_relu_4 = nn.Sequential()
        self.to_relu_5 = nn.Sequential()

        for x in range(4):
            self.to_relu_1.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2.add_module(str(x), features[x])
        for x in range(9, 18):
            self.to_relu_3.add_module(str(x), features[x])
        for x in range(18, 27):
            self.to_relu_4.add_module(str(x), features[x])
        for x in range(27, 36):
            self.to_relu_5.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1(x)
        h_relu_1 = h
        h = self.to_relu_2(h)
        h_relu_2 = h
        h = self.to_relu_3(h)
        h_relu_3 = h
        h = self.to_relu_4(h)
        h_relu_4 = h
        h = self.to_relu_5(h)
        h_relu_5 = h
        feature_map = (h_relu_1, h_relu_2, h_relu_3, h_relu_4, h_relu_5)
        content = [feature_map[idx] for idx in self.contentLayer]
        feature = [feature_map[idx] for idx in self.styleLayer]

        return feature, content
    
# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G