import torch.nn as nn
import torchvision.models as models

class ResNetBackboneModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackboneModel, self).__init__()
        # ResNet18 모델 로드 (pretrained=True로 사전 학습된 가중치 사용)
        self.backbone = models.resnet18(pretrained=pretrained)
        # 마지막 분류 레이어(fc)를 제거하고 feature extractor로 사용
        self.backbone.fc = nn.Identity()

        # 새로운 분류 레이어 추가 (이진 분류용)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # ResNet18의 출력 크기가 512
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 이진 분류를 위해 Sigmoid 사용
        )

    def forward(self, x):
        x = self.backbone(x)  # 백본을 통해 피처 추출
        x = self.classifier(x)  # 새로운 분류 레이어를 통과
        return x