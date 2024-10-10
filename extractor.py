import torch

class Extractor():
    def __init__(self, resnet, transform_func):
        if resnet == 50:
            model = 'resnet50'
        elif resnet == 18:
            model = 'resnet18'

        self.extractor = torch.hub.load('pytorch/vision:v0.10.0', model, pretrained=True)
        self.extractor.to('cuda')
        self.transform_func = transform_func

    def extract_multiple(self, images):
        input_batch = images.to('cuda')
        with torch.no_grad():
            features = self.extractor.conv1(input_batch)
            features = self.extractor.bn1(features)
            features = self.extractor.relu(features)
            features = self.extractor.maxpool(features)

            features = self.extractor.layer1(features)
            features = self.extractor.layer2(features)
            features = self.extractor.layer3(features)
            features = self.extractor.layer4(features)

            if self.transform_func == 'avg':
                features = torch.mean(features, dim=[2, 3]).squeeze()
            elif self.transform_func == 'flatten' or self.transform_func == 'custom':
                features = features.flatten(start_dim=1)
            
        return features