"""
Training-curve, ROC/PR, and F1-vs-threshold plots. Run after 06_train.py / 07_evaluate_test.py.

Original notebook cell(s): [15, 16, 17]
Depends on (run first / shares globals with): 06_train.py, 07_evaluate_test.py
"""

# ----- (notebook cell 15) -----
epochs = history['epoch']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (trnkey, valkey, name) in zip(axes, [
    ('trn_roc_auc', 'val_roc_auc', 'ROC-AUC'),
    ('trn_pr_auc',  'val_pr_auc',  'PR-AUC'),
    ('trn_f1',      'val_f1',      'F1-Score (thr 0.5)'),
]):
    ax.plot(epochs, history[trnkey], label=f'Train {name} (clean)', color='tab:blue', lw=2)
    ax.plot(epochs, history[valkey], label=f'Val {name}', color='tab:orange', lw=2)
    ax.set_title(f'{name} vs Epochs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel(name)
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
plt.tight_layout()
plt.savefig(CFG.OUTPUTDIR / 'metrics_progression.png', dpi=300)
plt.show()

# ----- (notebook cell 16) -----
val_fpr, val_tpr, _ = roc_curve(vallabels, valpreds)
tst_fpr, tst_tpr, _ = roc_curve(testlabels, testpreds)
val_prec_c, val_rec_c, _ = precision_recall_curve(vallabels, valpreds)
tst_prec_c, tst_rec_c, _ = precision_recall_curve(testlabels, testpreds)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(val_fpr, val_tpr, label=f'Validation ROC (AUC = {bestvalauc:.4f})', color='tab:orange', lw=2)
axes[0].plot(tst_fpr, tst_tpr, label=f'Test ROC (AUC = {tstauc:.4f})', color='tab:green', lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
axes[0].set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].grid(True, linestyle='--', alpha=0.6); axes[0].legend()

axes[1].plot(val_rec_c, val_prec_c, label=f'Validation PR (AP = {valprauc_best:.4f})', color='tab:orange', lw=2)
axes[1].plot(tst_rec_c, tst_prec_c, label=f'Test PR (AP = {tstprauc:.4f})', color='tab:green', lw=2)
axes[1].set_title('Precision-Recall Curves Comparison', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].grid(True, linestyle='--', alpha=0.6); axes[1].legend()

plt.tight_layout()
plt.savefig(CFG.OUTPUTDIR / 'roc_pr_curves_comparison.png', dpi=300)
plt.show()

# ----- (notebook cell 17) -----
thresh_range = np.linspace(0.01, 0.99, 100)
trnpredsnp, valpredsnp = np.array(trnpreds), np.array(valpreds)
trn_f1_curve = [f1_score(trnlabels,  (trnpredsnp  >= t).astype(int), zero_division=0) for t in thresh_range]
val_f1_curve = [f1_score(vallabels,  (valpredsnp  >= t).astype(int), zero_division=0) for t in thresh_range]
tst_f1_curve = [f1_score(testlabels, (testpredsnp >= t).astype(int), zero_division=0) for t in thresh_range]

plt.figure(figsize=(10, 6))
plt.plot(thresh_range, trn_f1_curve, label='Train F1', color='tab:blue', lw=2)
plt.plot(thresh_range, val_f1_curve, label='Validation F1', color='tab:orange', lw=2)
plt.plot(thresh_range, tst_f1_curve, label='Test F1', color='tab:green', lw=2)
plt.axvline(x=bestthresh, color='red', linestyle='--', lw=1.5,
            label=f'Optimal Val Threshold: {bestthresh:.2f}')
test_f1_at_opt = f1_score(testlabelsnp, binopt, zero_division=0)
plt.scatter([bestthresh], [test_f1_at_opt], color='red', s=50, zorder=5,
            label=f'Test F1 @ Opt: {test_f1_at_opt:.4f}')
plt.title('F1 Score vs Probability Threshold (Train vs Val vs Test)', fontsize=13, fontweight='bold')
plt.xlabel('Classification Threshold', fontsize=11); plt.ylabel('F1 Score', fontsize=11)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.02])
plt.grid(True, linestyle='--', alpha=0.6); plt.legend(loc='lower center', fontsize=10)
plt.tight_layout()
plt.savefig(CFG.OUTPUTDIR / '3in1_f1_threshold_curve.png', dpi=300)
plt.show()

gc.collect()
torch.cuda.empty_cache()
