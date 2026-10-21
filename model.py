"""ResNet18 and a documented formula-based FDPE-Net reconstruction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


MODEL_NAMES = ['baseline', 'fdpe', 'no_dfdm', 'no_hbff', 'no_physloss']


class DFDM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.AdaptiveAvgPool2d(1), nn.Sigmoid())
        self.low = nn.Parameter(torch.tensor(.8))
        self.high = nn.Parameter(torch.tensor(1.2))

    @staticmethod
    def distance(height, width, device):
        y = torch.fft.fftfreq(height, device=device)
        x = torch.fft.rfftfreq(width, device=device)
        distance = (y[:, None].square() + x[None, :].square()).sqrt()
        return distance / distance.max().clamp_min(1e-8)

    def forward(self, x):
        # Float32 FFT is deliberate: fp16 cuFFT has shape constraints and masks
        # need reliable STE gradients. Normalization matches torch's default.
        spectrum = torch.fft.rfft2(x.float())
        distance = self.distance(*x.shape[-2:], x.device)
        threshold = self.threshold(x.float())
        soft = torch.sigmoid(30 * (threshold - distance))
        hard = (distance <= threshold).float()
        mask = (hard - soft).detach() + soft
        enhanced = spectrum * (self.low * mask + self.high * (1 - mask))
        return x + torch.fft.irfft2(enhanced, s=x.shape[-2:]).to(x.dtype)


class FixedPhysics(nn.Module):
    """40 real Gabor magnitudes + normalized local contrast, all deterministic."""
    def __init__(self):
        super().__init__()
        grid = torch.arange(-20, 21, dtype=torch.float32)
        y, x = torch.meshgrid(grid, grid, indexing='ij')
        kernels = []
        for wavelength in [4, 6, 8, 10, 12]:
            for degree in range(0, 360, 45):
                theta = math.radians(degree)
                xp, yp = x * math.cos(theta) + y * math.sin(theta), -x * math.sin(theta) + y * math.cos(theta)
                sigma = .56 * wavelength
                kernel = torch.exp(-(xp.square() + .5 ** 2 * yp.square()) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * xp / wavelength)
                kernels.append(kernel)
        self.register_buffer('kernels', torch.stack(kernels).unsqueeze(1))

    @torch.no_grad()
    def forward(self, images):
        responses = F.conv2d(F.pad(images, (20, 20, 20, 20), mode='reflect'), self.kernels).abs()
        padded = F.pad(images, (3, 3, 3, 3), mode='reflect')
        high, low = F.max_pool2d(padded, 7, stride=1), -F.max_pool2d(-padded, 7, stride=1)
        contrast = (high - low) / (high + low + 1e-8)
        # Linear 1x1 channel reduction commutes with this spatial area resize.
        return F.interpolate(torch.cat([responses, contrast], 1), size=(16, 16), mode='area')


class PhysicsBranch(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.reduce = nn.Conv2d(40, 20, 1)
        self.initial = nn.Conv2d(21, dim, 3, padding=1)
        layers = []
        for _ in range(3):
            layers.extend([nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU()])
        self.refine = nn.Sequential(*layers)
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(dim, dim, 3, stride=4, padding=1)

    def forward(self, fixed):
        x = self.initial(torch.cat([self.reduce(fixed[:, :40]), fixed[:, 40:]], 1))
        x = x + self.refine(x)
        return [x, self.down2(x), self.down4(x)]


class CrossFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d_to_p = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.p_to_d = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.d_norm, self.p_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.compress = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, deep, physical):
        b, c, h, w = deep.shape
        d, p = deep.flatten(2).transpose(1, 2), physical.flatten(2).transpose(1, 2)
        enhanced_d = self.d_norm(d + self.d_to_p(d, p, p, need_weights=False)[0])
        enhanced_p = self.p_norm(p + self.p_to_d(p, d, d, need_weights=False)[0])
        merged = torch.cat([enhanced_d, enhanced_p], -1).transpose(1, 2).reshape(b, 2 * c, h, w)
        return self.compress(merged)


class HBFF(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in [128, 256, 512]])
        self.cross = nn.ModuleList([CrossFusion(dim) for _ in range(3)])
        self.smooth3, self.smooth2 = nn.Conv2d(dim, dim, 3, padding=1), nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, deep, physical):
        fused = [cross(projection(d), p) for cross, projection, d, p in zip(self.cross, self.projections, deep, physical)]
        middle = self.smooth3(fused[1] + F.interpolate(fused[2], size=fused[1].shape[-2:], mode='bilinear', align_corners=False))
        return self.smooth2(fused[0] + F.interpolate(middle, size=fused[0].shape[-2:], mode='bilinear', align_corners=False))


class RetrievalBackbone(nn.Module):
    """Standard ResNet18 stem and stages; new one-channel input, random weights."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2, self.layer3, self.layer4 = backbone.layer2, backbone.layer3, backbone.layer4

    def forward(self, image):
        x = self.stem(image)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c2, c3, c4]


class PaperModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError('unknown model: ' + name)
        self.name = name
        self.backbone = RetrievalBackbone()
        self.feature_dim = 512 if name == 'baseline' else 128
        self.head = nn.Linear(self.feature_dim, 10)
        if name != 'baseline':
            self.dfdm = DFDM(256) if name != 'no_dfdm' else nn.Identity()
            if name != 'no_hbff':
                self.fixed = FixedPhysics()
                self.physics = PhysicsBranch()
                self.hbff = HBFF()
            else:
                self.fallback = nn.Conv2d(256, 128, 1)
            # no_physloss also removes unused reconstruction parameters.
            if name != 'no_physloss':
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                    nn.ConvTranspose2d(128, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def encode(self, images, fixed_physics=None):
        deep = self.backbone(images)
        if self.name == 'baseline':
            enhanced = deep[-1]
        else:
            deep[1] = self.dfdm(deep[1])
            if self.name == 'no_hbff':
                enhanced = F.interpolate(self.fallback(deep[1]), size=(16, 16), mode='bilinear', align_corners=False)
            else:
                fixed_physics = self.fixed(images) if fixed_physics is None else fixed_physics
                enhanced = self.hbff(deep, self.physics(fixed_physics))
        return F.adaptive_avg_pool2d(enhanced, 1).flatten(1), enhanced

    def forward(self, images, fixed_physics=None):
        features, enhanced = self.encode(images, fixed_physics)
        output = dict(features=features, logits=self.head(features))
        if hasattr(self, 'decoder'):
            output['reconstruction'] = self.decoder(enhanced)
        return output


def build_model(name):
    return PaperModel(name)
