"""
Metric computation plus the single-epoch train/eval loop functions.

Original notebook cell(s): [12]
Depends on (run first / shares globals with): 00_config.py, 03_losses.py (criterion), 01_data.py
"""

# ----- (notebook cell 12) -----
def computemetrics(ytrue, ypred):
    auc   = roc_auc_score(ytrue, ypred) if len(set(ytrue)) > 1 else 0.0
    prauc = average_precision_score(ytrue, ypred) if len(set(ytrue)) > 1 else 0.0
    binpreds = [1 if p >= 0.5 else 0 for p in ypred]
    return (auc, prauc,
            precision_score(ytrue, binpreds, zero_division=0),
            recall_score(ytrue, binpreds, zero_division=0),
            f1_score(ytrue, binpreds, zero_division=0),
            accuracy_score(ytrue, binpreds))


def trainepoch(model, loader, optimizer, scaler):
    model.train()
    totalloss, seen = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        target = (labels * (1 - CFG.LABELSMOOTH) + 0.5 * CFG.LABELSMOOTH
                  if CFG.LABELSMOOTH > 0 else labels)

        if CFG.MIXUPPROB > 0 and random.random() < CFG.MIXUPPROB:
            lam = float(np.random.beta(CFG.MIXUPALPHA, CFG.MIXUPALPHA))
            perm = torch.randperm(inputs.size(0), device=DEVICE)
            inputs = lam * inputs + (1 - lam) * inputs[perm]
            target = lam * target + (1 - lam) * target[perm]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=CFG.USEAMP):
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        totalloss += loss.item() * bs
        seen += bs
    return totalloss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    totalloss, seen, alllabels, allpreds = 0.0, 0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=CFG.USEAMP):
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)
        bs = labels.size(0)
        totalloss += loss.item() * bs
        seen += bs
        allpreds.extend(np.nan_to_num(torch.sigmoid(logits).float().cpu().numpy(), nan=0.5))
        alllabels.extend(labels.cpu().numpy())
    return (totalloss / max(seen, 1), *computemetrics(alllabels, allpreds), alllabels, allpreds)
