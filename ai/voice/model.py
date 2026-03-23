# ─────────────────────────────────────────────────────────────
# Voice Emotion LSTM Model Definition
# Input : sequence of MFCC features  [batch, time, 40]
# Output: 5 emotion logits
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn


class VoiceModel(nn.Module):
    """Bidirectional LSTM for speech emotion recognition."""

    def __init__(self, n_mfcc: int = 40, hidden: int = 128, num_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attn = nn.Linear(hidden * 2, 1)  # attention over time steps
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                    # [B, T, 2H]
        weights = torch.softmax(self.attn(out), dim=1)  # [B, T, 1]
        ctx = (weights * out).sum(dim=1)         # [B, 2H]
        return self.classifier(ctx)


if __name__ == "__main__":
    model = VoiceModel()
    sample = torch.randn(2, 100, 40)             # batch=2, 100 frames, 40 MFCCs
    print("Output shape:", model(sample).shape)  # [2, 5]
