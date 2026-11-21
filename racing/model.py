"""Compact multi-horizon CNN; no pretrained download, no training on import."""
import torch
from torch import nn


class RacingCNN(nn.Module):
    """model.temporal = "stack" (default): the frames are stacked as image channels.
    model.temporal = "lstm": the same CNN encodes each frame separately, then an LSTM
    reads the per-frame features in time order (oldest first). The input format,
    (batch, 3*frames, H, W), is identical in both cases."""
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.frames = m["frames"]
        self.temporal = m.get("temporal", "stack")
        self.height = cfg["preprocess"]["height"]
        self.width = cfg["preprocess"]["width"]
        layers, incoming = [], 3*self.frames if self.temporal == "stack" else 3
        for i, outgoing in enumerate(m["channels"]):
            kernel, stride = (5, 2) if i < 3 else (3, 1)
            layers.extend([nn.Conv2d(incoming, outgoing, kernel, stride, kernel//2), nn.ELU()])
            incoming = outgoing
        # Preserve coarse spatial position; global pooling would lose lane offset.
        self.features = nn.Sequential(*layers, nn.AvgPool2d((self.height//24, self.width//40)), nn.Flatten())
        features = incoming*15
        if self.temporal == "lstm":
            self.lstm = nn.LSTM(features, m.get("lstm_hidden", 128), batch_first=True)
            features = m.get("lstm_hidden", 128)
        layers, incoming = [], features + 1
        for outgoing in m["dense"]:
            layers.extend([nn.Linear(incoming, outgoing), nn.ELU(), nn.Dropout(m["dropout"])])
            incoming = outgoing
        self.head = nn.Sequential(*layers, nn.Linear(incoming, len(m["horizons_s"])+int(m.get("throttle_output", False))), nn.Tanh())

    def forward(self, image, speed):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3*self.frames, self.height, self.width):
            raise ValueError("Incorrect CNN image dimensions")
        if speed.shape != (image.shape[0], 1):
            raise ValueError("Speed must have shape (batch, 1)")
        if self.temporal == "lstm":
            b = image.shape[0]
            per_frame = self.features(image.reshape(b*self.frames, 3, self.height, self.width))
            sequence, _ = self.lstm(per_frame.reshape(b, self.frames, -1))
            encoded = sequence[:, -1]
        else:
            encoded = self.features(image)
        return self.head(torch.cat((encoded, speed), dim=1))
