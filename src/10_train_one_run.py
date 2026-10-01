"""
Reusable `train_one_run()` for ablations / learning-curve experiments (trains a fresh model end to end on a given train/val/test split).

Original notebook cell(s): [20]
Depends on (run first / shares globals with): 00_config.py, 01_data.py, 02_model.py, 03_losses.py, 04_optim_utils.py
"""

# ----- (notebook cell 20) -----
def train_one_run(df_train, df_val, df_test, run_seed=42, run_tag='lc'):
    """Full training run from scratch. Returns dict of metrics."""
    setseed(run_seed)

    # ---------- Cache ----------
    s_tr, l_tr, y_tr = buildrawcache(df_train, f'{run_tag}-tr')
    tr_cache = finalizecache(s_tr, l_tr, y_tr, LIDARMEAN, LIDARSTD)
    del s_tr, l_tr; gc.collect()

    s_v, l_v, y_v = buildrawcache(df_val, f'{run_tag}-va')
    va_cache = finalizecache(s_v, l_v, y_v, LIDARMEAN, LIDARSTD)
    del s_v, l_v; gc.collect()

    s_t, l_t, y_t = buildrawcache(df_test, f'{run_tag}-te')
    te_cache = finalizecache(s_t, l_t, y_t, LIDARMEAN, LIDARSTD)
    del s_t, l_t; gc.collect()

    # ---------- Sampler ----------
    cnt = df_train['label'].value_counts().to_dict()
    if cnt.get(0, 0) == 0 or cnt.get(1, 0) == 0:
        print(f'  [{run_tag}] SKIP — degenerate subset '
              f'(pos={cnt.get(1,0)}, neg={cnt.get(0,0)})')
        return {'run_tag': run_tag, 'error': 'degenerate',
                'n_train': len(df_train), 'frac': None}

    weights = [1.0 / cnt[lbl] for lbl in df_train['label']]
    smp = WeightedRandomSampler(weights=weights,
                                num_samples=len(df_train),
                                replacement=True)

    trn_ld    = DataLoader(MultiModalDataset(tr_cache, True),  batch_size=CFG.BATCHSIZE,
                           sampler=smp, num_workers=CFG.NUMWORKERS,
                           pin_memory=True, drop_last=True)
    trn_ld_ev = DataLoader(MultiModalDataset(tr_cache, False), batch_size=CFG.BATCHSIZE * 2,
                           shuffle=False, num_workers=CFG.NUMWORKERS, pin_memory=True)
    va_ld     = DataLoader(MultiModalDataset(va_cache, False), batch_size=CFG.BATCHSIZE * 2,
                           shuffle=False, num_workers=CFG.NUMWORKERS, pin_memory=True)
    te_ld     = DataLoader(MultiModalDataset(te_cache, False), batch_size=CFG.BATCHSIZE * 2,
                           shuffle=False, num_workers=CFG.NUMWORKERS, pin_memory=True)

    # ---------- Model / optim / sched / loss ----------
    mdl = createconvnextmodel(CFG.INCHANNELS, CFG.NUMCLASSES, CFG.DROPOUT).to(DEVICE)
    freezeearlystages(mdl, CFG.FREEZESTAGES)
    setbackbonefrozen(mdl, frozen=True)

    opt = optim.AdamW(buildparamgroups(mdl, CFG.BASELR, CFG.WEIGHTDECAY,
                                       CFG.HEADLRMULT, CFG.LLRD))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                               factor=CFG.SCHED_FACTOR,
                                               patience=CFG.SCHED_PATIENCE,
                                               min_lr=1e-6)
    scl = torch.amp.GradScaler(device=DEVICE.type, enabled=CFG.USEAMP)

    pos_weight_t = None
    if CFG.USEPOSWEIGHT:
        pos_weight_t = torch.tensor([cnt.get(0, 1) / max(cnt.get(1, 1), 1)],
                                    dtype=torch.float32).to(DEVICE)
    crit = (FocalLoss(gamma=CFG.FOCALGAMMA, pos_weight=pos_weight_t)
            if CFG.USEFOCALLOSS else nn.BCEWithLogitsLoss(pos_weight=pos_weight_t))

    def _train_ep(m, ld, o, sc):
        m.train()
        tl, seen = 0.0, 0
        for x, y in ld:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            tgt = (y * (1 - CFG.LABELSMOOTH) + 0.5 * CFG.LABELSMOOTH) \
                  if CFG.LABELSMOOTH > 0 else y
            if CFG.MIXUPPROB > 0 and random.random() < CFG.MIXUPPROB:
                lam = float(np.random.beta(CFG.MIXUPALPHA, CFG.MIXUPALPHA))
                perm = torch.randperm(x.size(0), device=DEVICE)
                x = lam * x + (1 - lam) * x[perm]
                tgt = lam * tgt + (1 - lam) * tgt[perm]
            o.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=CFG.USEAMP):
                lg = m(x).squeeze(1)
                ls = crit(lg, tgt)
            sc.scale(ls).backward()
            sc.unscale_(o)
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            sc.step(o)
            sc.update()
            tl += ls.item() * x.size(0)
            seen += x.size(0)
        return tl / max(seen, 1)

    @torch.no_grad()
    def _eval(m, ld):
        m.eval()
        tl, seen, ys, ps = 0.0, 0, [], []
        for x, y in ld:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=CFG.USEAMP):
                lg = m(x).squeeze(1)
                ls = crit(lg, y)
            tl += ls.item() * x.size(0)
            seen += x.size(0)
            ps.extend(np.nan_to_num(torch.sigmoid(lg).float().cpu().numpy(),
                                    nan=0.5))
            ys.extend(y.cpu().numpy())
        return tl / max(seen, 1), ys, ps

    # ---------- Train ----------
    ckpt = CFG.OUTPUTDIR / f'{run_tag}_best.pth'
    best_auc = 0.0
    no_imp = 0
    frozen = True

    for ep in range(1, CFG.NUMEPOCHS + 1):
        if frozen and ep > CFG.FREEZEEPOCHS:
            setbackbonefrozen(mdl, frozen=False)
            frozen = False
            no_imp = 0

        _train_ep(mdl, trn_ld, opt, scl)
        _, vy, vp = _eval(mdl, va_ld)
        vauc = roc_auc_score(vy, vp) if len(set(vy)) > 1 else 0.0

        if not frozen:
            sch.step(vauc)

        if vauc > best_auc + CFG.MINDELTA:
            best_auc = vauc
            no_imp = 0
            torch.save(mdl.state_dict(), ckpt)
        else:
            if not frozen:
                no_imp += 1

        if ep % 5 == 0 or ep == 1:
            print(f'  [{run_tag}] ep {ep:02d} | '
                  f'val ROC-AUC {vauc:.4f} | best {best_auc:.4f}')

        if not frozen and no_imp >= CFG.PATIENCE:
            print(f'  [{run_tag}] early stop at ep {ep}')
            break

    # ---------- Final eval ----------
    mdl.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    mdl.eval()
    _, vy, vp = _eval(mdl, va_ld)
    _, ty, tp = _eval(mdl, te_ld)

    v_pr  = average_precision_score(vy, vp) if len(set(vy)) > 1 else 0.0
    v_roc = roc_auc_score(vy, vp)           if len(set(vy)) > 1 else 0.5
    t_pr  = average_precision_score(ty, tp) if len(set(ty)) > 1 else 0.0
    t_roc = roc_auc_score(ty, tp)           if len(set(ty)) > 1 else 0.5

    p_c, r_c, t_c = precision_recall_curve(vy, vp)
    f1c = 2 * (p_c[:-1] * r_c[:-1]) / (p_c[:-1] + r_c[:-1] + 1e-8)
    best_t = float(t_c[np.argmax(f1c)]) if len(t_c) else 0.5

    t_bin  = (np.array(tp) >= best_t).astype(int)
    t_f1   = f1_score(ty, t_bin, zero_division=0)
    t_prec = precision_score(ty, t_bin, zero_division=0)
    t_rec  = recall_score(ty, t_bin, zero_division=0)

    print(f'  [{run_tag}] DONE | n={len(df_train)} | '
          f'test PR-AUC={t_pr:.4f} ROC-AUC={t_roc:.4f} F1={t_f1:.4f}')

    return {
        'run_tag': run_tag,
        'n_train': len(df_train),
        'n_pos': int(df_train['label'].sum()),
        'n_neg': int((df_train['label'] == 0).sum()),
        'val_pr_auc': v_pr,
        'val_roc_auc': v_roc,
        'test_pr_auc': t_pr,
        'test_roc_auc': t_roc,
        'test_f1': t_f1,
        'test_precision': t_prec,
        'test_recall': t_rec,
        'threshold': best_t,
    }
