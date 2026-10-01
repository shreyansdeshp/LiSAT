"""
Backbone freezing/unfreezing and discriminative (layer-wise decayed) learning-rate param groups.

Original notebook cell(s): [10]
Depends on (run first / shares globals with): 00_config.py
"""

# ----- (notebook cell 10) -----
def ishead(name: str) -> bool:
    return name.startswith('features.0.0') or name.startswith('features.0.1') or name.startswith('classifier')

def stagedepth(name: str) -> int:
    if name.startswith('classifier'): return 8
    if name.startswith('features.'): return int(name.split('.')[1])
    return 0

def freezeearlystages(model, upto: int):
    for name, p in model.named_parameters():
        if ishead(name):
            p.requires_grad = True
            continue
        if name.startswith('features.'):
            if int(name.split('.')[1]) < upto:
                p.requires_grad = False

def setbackbonefrozen(model, frozen: bool):
    for name, p in model.named_parameters():
        if ishead(name):
            p.requires_grad = True
            continue
        if name.startswith('features.'):
            stage = int(name.split('.')[1])
            if stage < CFG.FREEZESTAGES:
                continue
        p.requires_grad = not frozen

def buildparamgroups(model, baselr, weightdecay, headlrmult, llrd):
    maxdepth = 8
    buckets = {}
    for name, p in model.named_parameters():
        lr = baselr * headlrmult if ishead(name) else baselr * (llrd ** (maxdepth - stagedepth(name)))
        wd = 0.0 if (p.ndim <= 1 or 'norm' in name.lower() or 'layer_scale' in name.lower()) else weightdecay
        buckets.setdefault((round(lr, 12), wd), []).append((name, p))

    groups = []
    for (lr, wd), items in sorted(buckets.items()):
        groups.append({'params': [p for _, p in items], 'lr': lr, 'weight_decay': wd})
    return groups
