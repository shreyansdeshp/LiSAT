"""
Loads the best checkpoint and reports the final benchmark on the untouched test set.

Original notebook cell(s): [14]
Depends on (run first / shares globals with): 06_train.py
"""

# ----- (notebook cell 14) -----
if ckptpath.exists() and bestvalauc > 0.0:
    model.load_state_dict(torch.load(ckptpath, map_location=DEVICE, weights_only=True))
    model.eval()

    _, _, _, _, _, _, _, trnlabels, trnpreds  = evaluate(model, trnloader_eval)
    _, _, _, _, _, _, _, vallabels, valpreds  = evaluate(model, valloader)
    _, tstauc, tstprauc, _, _, _, _, testlabels, testpreds = evaluate(model, testloader)

    valprauc_best = average_precision_score(vallabels, valpreds)
    valprecisions, valrecalls, valthresholds = precision_recall_curve(vallabels, valpreds)
    f1_scores = 2 * (valprecisions[:-1] * valrecalls[:-1]) / (valprecisions[:-1] + valrecalls[:-1] + 1e-8)
    bestthresh = float(valthresholds[np.argmax(f1_scores)])

    testpredsnp, testlabelsnp = np.array(testpreds), np.array(testlabels)
    bindefault = (testpredsnp >= 0.5).astype(int)
    binopt     = (testpredsnp >= bestthresh).astype(int)

    print("\n" + "=" * 68 + "\n      FINAL BENCHMARK: UNTOUCHED TEST SET (CONVNEXT-TINY)\n" + "=" * 68)
    print(f"Threshold-free   -> ROC-AUC: {tstauc:.4f} | PR-AUC: {tstprauc:.4f}")
    print(f"Optimal threshold calibrated on val: {bestthresh:.4f}")

    for tag, binp in [("Default (0.50)", bindefault), (f"Calibrated ({bestthresh:.4f})", binopt)]:
        print(f"\nTest @ {tag}:")
        print(f"  • Precision : {precision_score(testlabelsnp, binp, zero_division=0):.4f}")
        print(f"  • Recall    : {recall_score(testlabelsnp, binp, zero_division=0):.4f}")
        print(f"  • F1-Score  : {f1_score(testlabelsnp, binp, zero_division=0):.4f}")
        print(f"  • Accuracy  : {accuracy_score(testlabelsnp, binp):.4f}")
    print("=" * 68)
