"""
Main training entrypoint: builds caches/loaders, model, optimizer, scheduler, and runs the training loop with early stopping and checkpointing. Run this first.

Original notebook cell(s): [6, 7, 11, 13]
Depends on (run first / shares globals with): 00_config.py, 01_data.py, 02_model.py, 03_losses.py, 04_optim_utils.py, 05_engine.py
"""

# ----- (notebook cell 6) -----
trnsats, trnlidars, trnlabels_raw = buildrawcache(traindf, "Train")
LIDARMEAN, LIDARSTD = computelidarstats(trnlidars)
print(f"LiDAR channel means: {np.round(LIDARMEAN, 3)}")
print(f"LiDAR channel stds : {np.round(LIDARSTD, 3)}")

traincache = finalizecache(trnsats, trnlidars, trnlabels_raw, LIDARMEAN, LIDARSTD)
del trnsats, trnlidars; gc.collect()

vs, vl, vlab = buildrawcache(valdf, "Val")
valcache = finalizecache(vs, vl, vlab, LIDARMEAN, LIDARSTD)
del vs, vl; gc.collect()

ts, tl, tlab = buildrawcache(testdf, "Test")
testcache = finalizecache(ts, tl, tlab, LIDARMEAN, LIDARSTD)
del ts, tl; gc.collect()

# ----- (notebook cell 7) -----
counts = traindf['label'].value_counts().to_dict()
sampleweights = [1.0 / counts[lbl] for lbl in traindf['label']]
sampler = WeightedRandomSampler(weights=sampleweights, num_samples=len(traindf), replacement=True)

trnloader      = DataLoader(MultiModalDataset(traincache, True),  batch_size=CFG.BATCHSIZE,
                            sampler=sampler, num_workers=CFG.NUMWORKERS, pin_memory=True, drop_last=True)
trnloader_eval = DataLoader(MultiModalDataset(traincache, False), batch_size=CFG.BATCHSIZE * 2,
                            shuffle=False, num_workers=CFG.NUMWORKERS, pin_memory=True)
valloader      = DataLoader(MultiModalDataset(valcache, False),    batch_size=CFG.BATCHSIZE * 2,
                            shuffle=False, num_workers=CFG.NUMWORKERS, pin_memory=True)
testloader     = DataLoader(MultiModalDataset(testcache, False),   batch_size=CFG.BATCHSIZE * 2,
                            shuffle=False, num_workers=CFG.NUMWORKERS, pin_memory=True)

# ----- (notebook cell 11) -----
model = createconvnextmodel(CFG.INCHANNELS, CFG.NUMCLASSES, CFG.DROPOUT).to(DEVICE)
freezeearlystages(model, CFG.FREEZESTAGES)
setbackbonefrozen(model, frozen=True)

optimizer = optim.AdamW(buildparamgroups(model, CFG.BASELR, CFG.WEIGHTDECAY, CFG.HEADLRMULT, CFG.LLRD))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=CFG.SCHED_FACTOR,
                                                 patience=CFG.SCHED_PATIENCE, min_lr=1e-6)
scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=CFG.USEAMP)
backbonefrozen = True

# ----- (notebook cell 13) -----
bestvalauc = 0.0
epochsnoimprove = 0
ckptpath = CFG.OUTPUTDIR / 'convnext_tiny_multimodal_best.pth'

history = {'epoch': [], 'trn_loss': [], 'val_loss': [],
           'trn_roc_auc': [], 'val_roc_auc': [],
           'trn_pr_auc': [], 'val_pr_auc': [],
           'trn_f1': [], 'val_f1': []}

print(f'\n{"="*78}\n  ConvNeXt-Tiny | 11-channel multimodal | max {CFG.NUMEPOCHS} epochs, patience {CFG.PATIENCE}\n{"="*78}')

for epoch in range(1, CFG.NUMEPOCHS + 1):
    if backbonefrozen and epoch > CFG.FREEZEEPOCHS:
        setbackbonefrozen(model, frozen=False)
        backbonefrozen = False
        epochsnoimprove = 0
        print(f'  -> Unfroze backbone at epoch {epoch}')

    trnloss = trainepoch(model, trnloader, optimizer, scaler)
    valloss, valauc, valprauc, valprec, valrec, valf1, valacc, _, _ = evaluate(model, valloader)

    if not backbonefrozen:
        scheduler.step(valauc)

    if CFG.EVALTRAINEACHEPOCH:
        ctrnloss, trnauc, trnprauc, _, _, trnf1, _, _, _ = evaluate(model, trnloader_eval)
    else:
        ctrnloss, trnauc, trnprauc, trnf1 = trnloss, 0.0, 0.0, 0.0

    history['epoch'].append(epoch)
    history['trn_loss'].append(ctrnloss)
    history['val_loss'].append(valloss)
    history['trn_roc_auc'].append(trnauc)
    history['val_roc_auc'].append(valauc)
    history['trn_pr_auc'].append(trnprauc)
    history['val_pr_auc'].append(valprauc)
    history['trn_f1'].append(trnf1)
    history['val_f1'].append(valf1)

    saved = ''
    if valauc > bestvalauc + CFG.MINDELTA:
        bestvalauc = valauc
        epochsnoimprove = 0
        torch.save(model.state_dict(), ckptpath)
        saved = '  ✓ saved'
    else:
        if not backbonefrozen:
            epochsnoimprove += 1

    headlr = max(g['lr'] for g in optimizer.param_groups)
    print(f'Ep {epoch:02d}/{CFG.NUMEPOCHS} | head LR: {headlr:.2e} | '
          f'clean AUC gap: {trnauc - valauc:+.4f} | patience: {epochsnoimprove}/{CFG.PATIENCE}{saved}')
    print(f'  Train (aug)   -> Loss: {trnloss:.4f}')
    print(f'  Train (clean) -> Loss: {ctrnloss:.4f} | ROC-AUC: {trnauc:.4f} | PR-AUC: {trnprauc:.4f} | F1: {trnf1:.4f}')
    print(f'  Val           -> Loss: {valloss:.4f} | ROC-AUC: {valauc:.4f} | PR-AUC: {valprauc:.4f} | F1: {valf1:.4f}')

    if not backbonefrozen and epochsnoimprove >= CFG.PATIENCE:
        print(f'\nEarly stopping at epoch {epoch}: val ROC-AUC flat for {CFG.PATIENCE} epochs.')
        break
