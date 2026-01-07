from sklearn.metrics import roc_auc_score,roc_curve
import uproot,os,sys
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
hep.style.use(hep.style.CMS)

def detect_binary_type(output_dir):
    """Auto-detect which binary classification based on files in directory."""
    files = os.listdir(output_dir)
    
    if any('WZ_vs_QCD' in f for f in files):
        return 'WZ_vs_QCD', ['WToQQ', 'ZToQQ', 'WZ'], 'score_(label_Wqq) | (label_Zqq)'
    elif any('HToCC_vs_QCD' in f for f in files):
        return 'HToCC_vs_QCD', ['HToCC'], 'score_label_Hcc'
    elif any('TTbar_vs_QCD' in f for f in files):
        return 'TTbar_vs_QCD', ['TTBar'], 'score_label_Tbqq'
    else:
        raise ValueError("Cannot detect binary classification type from files")

def auc_calc_binary(output_dir: str, binary_type: str, process: str, score_branch: str):
    """
    Binary S vs QCD evaluation exactly as defined in the ParT paper.
    Computes AUC and background rejection at the correct TPR.
    """

    # Use 50% TPR for all processes
    tpr_target = 0.50

    # ---------------------------
    # Load SIGNAL predictions
    # ---------------------------
    if process == 'WZ':
        # Combined W+Z signal
        sig_scores_list = []
        for sig_process in ['WToQQ', 'ZToQQ']:
            sig_file = os.path.join(output_dir, f"pred_{binary_type}_{sig_process}.root")
            with uproot.open(sig_file)["Events"] as events:
                pS = events[score_branch].array(library="np")
                pB = events["score_label_QCD"].array(library="np")
                sig_scores_list.append(pS / (pS + pB))
        sig_scores = np.concatenate(sig_scores_list)
        sig_labels = np.ones_like(sig_scores)
    else:
        # Individual signal process
        sig_file = os.path.join(output_dir, f"pred_{binary_type}_{process}.root")
        with uproot.open(sig_file)["Events"] as events:
            pS = events[score_branch].array(library="np")
            pB = events["score_label_QCD"].array(library="np")
            sig_scores = pS / (pS + pB)
            sig_labels = np.ones_like(sig_scores)
    
    # ---------------------------
    # Load QCD BACKGROUND
    # ---------------------------
    qcd_file = os.path.join(output_dir, f"pred_{binary_type}_QCD.root")
    with uproot.open(qcd_file)["Events"] as events:
        pS = events[score_branch].array(library="np")
        pB = events["score_label_QCD"].array(library="np")
        bkg_scores = pS / (pS + pB)
        bkg_labels = np.zeros_like(bkg_scores)

    # Combine into a binary dataset
    scores = np.concatenate([bkg_scores, sig_scores])
    labels = np.concatenate([bkg_labels, sig_labels])

    # Compute ROC (binary): pass positive-class probabilities
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)

    # Prepare rejection values (1 / FPR) and avoid division-by-zero in plotting
    rej_curve = np.divide(1.0, fpr, out=np.full_like(fpr, np.nan), where=fpr > 0)

    # Evaluate rejection at the CORRECT efficiency using interpolation
    # Find FPR at the target TPR, then rejection = 1/FPR
    fpr_at_target = np.interp(tpr_target, tpr, fpr)
    rej_at_target = np.inf if fpr_at_target == 0 else 1.0 / fpr_at_target

    print(f"\nProcess: {process}")
    print(f"AUC = {auc:.4f}")
    print(f"Rejection at {tpr_target*100:.1f}% TPR = {rej_at_target:.1f}")

    # Plot ROC curve: TPR vs Rejection (FPR)
    plt.figure(figsize=(10, 8))
    plt.plot(tpr, fpr, label=f'{process} (AUC={auc:.4f}, Rej@50%={rej_at_target:.0f})', linewidth=2)
    #plt.yscale('log')
    plt.xlabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    plt.ylabel('Background Rejection (FPR)', fontsize=12)
    plt.title(f'ROC Curve: {process} vs QCD', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'roc_{process}_binary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {os.path.join(output_dir, "plots", f"roc_{process}_binary.png")}')

    return auc, rej_at_target, fpr, tpr

    
if __name__ == '__main__':
    out_dir = sys.argv[1]
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    
    # Auto-detect binary classification type
    binary_type, processes, score_branch = detect_binary_type(out_dir)
    print(f"Detected binary type: {binary_type}")
    print(f"Processing: {processes}")
    print(f"Score branch: {score_branch}")
    
    # Process each signal against QCD
    for process in processes:
        try:
            auc_calc_binary(output_dir=out_dir, binary_type=binary_type, 
                          process=process, score_branch=score_branch)
        except Exception as e:
            print(f"Error processing {process}: {e}")
            import traceback
            traceback.print_exc()
            continue
