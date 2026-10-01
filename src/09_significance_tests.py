"""
Permutation test and bootstrap 95% confidence intervals on the test-set metrics.

Original notebook cell(s): [18, 19]
Depends on (run first / shares globals with): 07_evaluate_test.py
"""

# ----- (notebook cell 18) -----
# ==========================================
# PERMUTATION TEST — Is the model better than chance?
# ==========================================
# H0: predictions are independent of labels.
# Shuffle test labels N times, recompute metric, compare.
# p-value = fraction of null metrics >= observed.

print('\n' + '=' * 68)
print('  PERMUTATION TEST (H0: predictions independent of labels)')
print('=' * 68)

N_PERM = 2000
rng = np.random.default_rng(CFG.SEED)

testpredsnp  = np.asarray(testpreds,  dtype=np.float64)
testlabelsnp = np.asarray(testlabels, dtype=np.int64)

obs_pr  = average_precision_score(testlabelsnp, testpredsnp)
obs_roc = roc_auc_score(testlabelsnp, testpredsnp)

null_pr  = np.empty(N_PERM, dtype=np.float64)
null_roc = np.empty(N_PERM, dtype=np.float64)

for i in range(N_PERM):
    shuf = rng.permutation(testlabelsnp)
    if len(np.unique(shuf)) < 2:
        null_pr[i], null_roc[i] = 0.0, 0.5
        continue
    null_pr[i]  = average_precision_score(shuf, testpredsnp)
    null_roc[i] = roc_auc_score(shuf, testpredsnp)

# +1 correction so p is never exactly zero
p_pr  = (np.sum(null_pr  >= obs_pr)  + 1) / (N_PERM + 1)
p_roc = (np.sum(null_roc >= obs_roc) + 1) / (N_PERM + 1)

z_pr  = (obs_pr  - null_pr.mean())  / (null_pr.std()  + 1e-12)
z_roc = (obs_roc - null_roc.mean()) / (null_roc.std() + 1e-12)

print(f'Permutations: {N_PERM}')
print(f'Test class balance: {testlabelsnp.mean():.4f} positive '
      f'({testlabelsnp.sum()}/{len(testlabelsnp)})')
print()
print(f'{"Metric":<10} {"Observed":>10} {"Null mean":>12} {"Null std":>12} '
      f'{"Z-score":>10} {"p-value":>12}')
print('-' * 70)
print(f'{"PR-AUC":<10} {obs_pr:>10.4f} {null_pr.mean():>12.4f} '
      f'{null_pr.std():>12.4f} {z_pr:>10.3f} {p_pr:>12.6f}')
print(f'{"ROC-AUC":<10} {obs_roc:>10.4f} {null_roc.mean():>12.4f} '
      f'{null_roc.std():>12.4f} {z_roc:>10.3f} {p_roc:>12.6f}')
print()

if p_pr < 0.001:
    print('  -> PR-AUC significantly better than chance (p < 0.001).')
elif p_pr < 0.05:
    print('  -> PR-AUC significantly better than chance (p < 0.05).')
else:
    print('  -> PR-AUC NOT significant at p < 0.05.')

if p_roc < 0.001:
    print('  -> ROC-AUC significantly better than chance (p < 0.001).')
elif p_roc < 0.05:
    print('  -> ROC-AUC significantly better than chance (p < 0.05).')
else:
    print('  -> ROC-AUC NOT significant at p < 0.05.')

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(null_pr, bins=50, color='steelblue', alpha=0.75,
        label=f'Null distribution (n={N_PERM})')
ax.axvline(obs_pr, color='red', lw=2.5, label=f'Observed = {obs_pr:.4f}')
ax.axvline(null_pr.mean(), color='black', lw=1.2, ls='--',
           label=f'Null mean = {null_pr.mean():.4f}')
ax.set_title(f'Permutation test — PR-AUC\np = {p_pr:.5f}  |  z = {z_pr:.2f}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('PR-AUC under label shuffling')
ax.set_ylabel('Count')
ax.grid(True, ls='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=9)

ax = axes[1]
ax.hist(null_roc, bins=50, color='seagreen', alpha=0.75,
        label=f'Null distribution (n={N_PERM})')
ax.axvline(obs_roc, color='red', lw=2.5, label=f'Observed = {obs_roc:.4f}')
ax.axvline(null_roc.mean(), color='black', lw=1.2, ls='--',
           label=f'Null mean = {null_roc.mean():.4f}')
ax.set_title(f'Permutation test — ROC-AUC\np = {p_roc:.5f}  |  z = {z_roc:.2f}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('ROC-AUC under label shuffling')
ax.set_ylabel('Count')
ax.grid(True, ls='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(CFG.OUTPUTDIR / 'permutation_test.png', dpi=300)
plt.show()

np.savez(CFG.OUTPUTDIR / 'permutation_null_distributions.npz',
         null_pr=null_pr, null_roc=null_roc,
         obs_pr=obs_pr, obs_roc=obs_roc,
         p_pr=p_pr, p_roc=p_roc, n_perm=N_PERM)
print('Saved: permutation_null_distributions.npz')
print('=' * 68)

# ----- (notebook cell 19) -----
# ==========================================
# BOOTSTRAP 95% CONFIDENCE INTERVALS
# ==========================================
# Resample the test set N times with replacement.
# Compute metric on each resample. Report the 2.5 / 50 / 97.5 quantiles.

def bootstrap_ci(y_true, y_pred, metric_fn, n=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        yt = y_true[idx]
        yp = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, yp))
    scores = np.array(scores)
    return (np.quantile(scores, alpha / 2),
            np.quantile(scores, 0.5),
            np.quantile(scores, 1 - alpha / 2),
            scores)

print('\n' + '=' * 68)
print('  BOOTSTRAP 95% CONFIDENCE INTERVALS (n=2000)')
print('=' * 68)

# --- Threshold-free metrics ---
lo_pr,  med_pr,  hi_pr,  pr_scores  = bootstrap_ci(testlabelsnp, testpredsnp, average_precision_score)
lo_roc, med_roc, hi_roc, roc_scores = bootstrap_ci(testlabelsnp, testpredsnp, roc_auc_score)

# --- F1 at calibrated threshold ---
def f1_at_thr(yt, yp):
    return f1_score(yt, (np.asarray(yp) >= bestthresh).astype(int), zero_division=0)

lo_f1, med_f1, hi_f1, f1_scores = bootstrap_ci(testlabelsnp, testpredsnp, f1_at_thr)

# --- Precision and Recall at calibrated threshold ---
def prec_at_thr(yt, yp):
    return precision_score(yt, (np.asarray(yp) >= bestthresh).astype(int), zero_division=0)

def rec_at_thr(yt, yp):
    return recall_score(yt, (np.asarray(yp) >= bestthresh).astype(int), zero_division=0)

lo_prec, med_prec, hi_prec, _ = bootstrap_ci(testlabelsnp, testpredsnp, prec_at_thr)
lo_rec,  med_rec,  hi_rec,  _ = bootstrap_ci(testlabelsnp, testpredsnp, rec_at_thr)

print(f'{"Metric":<12} {"Point":>8} {"Median":>8} {"95% CI":>20}')
print('-' * 52)
print(f'{"ROC-AUC":<12} {obs_roc:>8.4f} {med_roc:>8.4f} '
      f'   [{lo_roc:.4f} – {hi_roc:.4f}]')
print(f'{"PR-AUC":<12} {obs_pr:>8.4f} {med_pr:>8.4f} '
      f'   [{lo_pr:.4f} – {hi_pr:.4f}]')
print(f'{"F1":<12} {"-":>8} {med_f1:>8.4f} '
      f'   [{lo_f1:.4f} – {hi_f1:.4f}]')
print(f'{"Precision":<12} {"-":>8} {med_prec:>8.4f} '
      f'   [{lo_prec:.4f} – {hi_prec:.4f}]')
print(f'{"Recall":<12} {"-":>8} {med_rec:>8.4f} '
      f'   [{lo_rec:.4f} – {hi_rec:.4f}]')
print('=' * 68)

# ---------- Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(roc_scores, bins=40, color='tab:orange', alpha=0.75)
ax.axvline(obs_roc, color='red', lw=2.5, label=f'Observed = {obs_roc:.4f}')
ax.axvline(lo_roc, color='black', lw=1, ls='--', label=f'95% CI: [{lo_roc:.3f}, {hi_roc:.3f}]')
ax.axvline(hi_roc, color='black', lw=1, ls='--')
ax.set_title('Bootstrap — ROC-AUC', fontweight='bold')
ax.set_xlabel('ROC-AUC'); ax.set_ylabel('Count')
ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

ax = axes[1]
ax.hist(pr_scores, bins=40, color='tab:blue', alpha=0.75)
ax.axvline(obs_pr, color='red', lw=2.5, label=f'Observed = {obs_pr:.4f}')
ax.axvline(lo_pr, color='black', lw=1, ls='--', label=f'95% CI: [{lo_pr:.3f}, {hi_pr:.3f}]')
ax.axvline(hi_pr, color='black', lw=1, ls='--')
ax.set_title('Bootstrap — PR-AUC', fontweight='bold')
ax.set_xlabel('PR-AUC'); ax.set_ylabel('Count')
ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

ax = axes[2]
ax.hist(f1_scores, bins=40, color='tab:green', alpha=0.75)
ax.axvline(med_f1, color='red', lw=2.5, label=f'Median = {med_f1:.4f}')
ax.axvline(lo_f1, color='black', lw=1, ls='--', label=f'95% CI: [{lo_f1:.3f}, {hi_f1:.3f}]')
ax.axvline(hi_f1, color='black', lw=1, ls='--')
ax.set_title('Bootstrap — F1 @ calibrated threshold', fontweight='bold')
ax.set_xlabel('F1'); ax.set_ylabel('Count')
ax.grid(True, ls='--', alpha=0.5); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(CFG.OUTPUTDIR / 'bootstrap_ci.png', dpi=300)
plt.show()

# Save arrays for the paper
np.savez(CFG.OUTPUTDIR / 'bootstrap_distributions.npz',
         pr_scores=pr_scores, roc_scores=roc_scores, f1_scores=f1_scores,
         obs_pr=obs_pr, obs_roc=obs_roc)
print('Saved: bootstrap_distributions.npz')
