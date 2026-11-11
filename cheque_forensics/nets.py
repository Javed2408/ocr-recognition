"""PyTorch network architectures. Training lives in training/, inference in the service modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(conv_bn(cin, cout), conv_bn(cout, cout))


class UNet(nn.Module):
    """Compact U-Net used for field segmentation, cheque boundary segmentation and
    tampered-pixel localisation (only the channel counts differ)."""

    def __init__(self, in_ch=3, out_ch=1, base=16, depth=4):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList([DoubleConv(in_ch, chans[0])] +
                                   [DoubleConv(chans[i], chans[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chans[i + 1], chans[i], 2, 2) for i in reversed(range(depth))])
        self.up_convs = nn.ModuleList([DoubleConv(chans[i] * 2, chans[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chans[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, conv in zip(self.ups, self.up_convs):
            skip = skips.pop()
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd input sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = conv(torch.cat([x, skip], 1))
        return self.head(x)


class CRNN(nn.Module):
    """CNN + BiLSTM text-line recogniser trained with CTC (class 0 is the CTC blank)."""

    def __init__(self, n_classes, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2),                 # 16 x W/2
            conv_bn(32, 48), nn.MaxPool2d(2),                # 8 x W/4
            conv_bn(48, 96), conv_bn(96, 96), nn.MaxPool2d((2, 1)),     # 4 x W/4
            conv_bn(96, 128), conv_bn(128, 128), nn.MaxPool2d((2, 1)),  # 2 x W/4
            conv_bn(128, 160, k=2), nn.AdaptiveAvgPool2d((1, None)),    # 1 x W/4
        )
        self.rnn = nn.LSTM(160, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):                    # x: (B, 1, 32, W)
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # (B, T, C)
        f, _ = self.rnn(f)
        return self.fc(f)                    # (B, T, n_classes) logits


class SignatureEmbedder(nn.Module):
    """One branch of the Siamese network: signature image -> L2-normalised embedding."""

    def __init__(self, dim=128):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 32, k=5, stride=2), nn.MaxPool2d(2),           # 24 x 48 (from 96 x 192)
            conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2),        # 12 x 24
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2),     # 6 x 12
            conv_bn(128, 192),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(192 * 2, 256), nn.ReLU(inplace=True), nn.Linear(256, dim))

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([F.adaptive_avg_pool2d(f, 1), F.adaptive_max_pool2d(f, 1)], 1).flatten(1)
        return F.normalize(self.head(f), dim=1)


class SiameseNetwork(nn.Module):
    """Two weight-sharing branches (the same SignatureEmbedder applied to both inputs)."""

    def __init__(self, dim=128):
        super().__init__()
        self.embedder = SignatureEmbedder(dim)

    def forward(self, a, b):
        ea, eb = self.embedder(a), self.embedder(b)
        return ea, eb, F.pairwise_distance(ea, eb)


def contrastive_loss(distance, same, margin=1.0):
    """Hadsell et al. contrastive loss. same=1 for genuine pairs, 0 for forged/other-writer pairs."""
    pos = same * distance.pow(2)
    neg = (1 - same) * F.relu(margin - distance).pow(2)
    return 0.5 * (pos + neg).mean()


class TamperClassifier(nn.Module):
    """Binary CNN: is this field window tampered? Keeps the last conv activations for Grad-CAM."""

    def __init__(self, in_ch=4):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(in_ch, 32), conv_bn(32, 32), nn.MaxPool2d(2),     # 32 x 128
            conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2),        # 16 x 64
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2),     # 8 x 32
            conv_bn(128, 192),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, 1))
        self.activations = None
        self.gradients = None

    def forward(self, x, keep_activations=False):
        a = self.features(x)
        if keep_activations:
            self.activations = a
            a.register_hook(self._save_grad)
        return self.classifier(F.adaptive_max_pool2d(a, 1).flatten(1)).squeeze(1)

    def _save_grad(self, grad):
        self.gradients = grad


class ConvAutoencoder(nn.Module):
    """Trained on genuine field windows only; high reconstruction error marks unusual regions."""

    def __init__(self, in_ch=4):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_bn(in_ch, 32, stride=2),   # 32 x 128
            conv_bn(32, 64, stride=2),      # 16 x 64
            conv_bn(64, 96, stride=2),      # 8 x 32
            conv_bn(96, 32),                # bottleneck: few channels so it cannot copy the input
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 96, 2, 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 2, 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
