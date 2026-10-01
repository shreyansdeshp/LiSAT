"""
Optional focal loss for class imbalance.

Original notebook cell(s): [9]
Depends on (run first / shares globals with): 00_config.py (CFG), 06_train.py (counts, DEVICE)
"""

# ----- (notebook cell 9) -----
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        ptcorrect = torch.where(targets >= 0.5, probs, 1 - probs)
        focalterm = (1 - ptcorrect).clamp(min=1e-6).pow(self.gamma)
        loss = focalterm * bce
        if self.pos_weight is not None:
            alphaweight = torch.where(targets >= 0.5, self.pos_weight, torch.ones_like(targets))
            loss = loss * alphaweight
        return loss.mean()

posweighttensor = None
if CFG.USEPOSWEIGHT:
    posweighttensor = torch.tensor([counts.get(0, 0) / max(counts.get(1, 1), 1)],
                                   dtype=torch.float32).to(DEVICE)

criterion = (FocalLoss(gamma=CFG.FOCALGAMMA, pos_weight=posweighttensor)
             if CFG.USEFOCALLOSS else nn.BCEWithLogitsLoss(pos_weight=posweighttensor))
