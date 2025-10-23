import numpy as np
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, log_loss
from utils import est_Cov


def calc_cov_sq_norm(Cov):
    """
    Calculate the covariance norm
    """
    
    # if shape is (d, k), then take the first k
    if Cov.ndim >= 2:
        
        d, k = Cov.shape
        sq_norm = np.sum(np.linalg.norm(Cov, ord=2, axis=0)**2)
    
    else:
        sq_norm = np.linalg.norm(Cov, ord=2)**2

    return sq_norm


def TPR(y, y_pred, binary=True):
    """
    Calculate the True Positive Rate (TPR)
    """

     # calculate the TPR - total
    correct = (y == y_pred)
    if binary:
        tpr = np.sum((correct & (y == 1))) / np.sum(y == 1)
    else:
        # check if only a single label is present
        if len(np.unique(y)) == 1:
            tpr = np.sum(correct)/ len(y)
        else:
            # go over each category and calculate the TPR
            all_tpr = []
            for label in np.unique(y):
            
                tpr_label = np.sum(correct & (y == label)) / np.sum(y == label)
                all_tpr.append(tpr_label)
            
            # take the mean of the TPRs
            tpr = np.mean(all_tpr)
    
    return tpr


def TPR_gap(y, y_pred, z, binary=True):
    """
    Calculate the True Positive Rate (TPR) gap between groups
    """

    # calculate TPR gap
    tpr_per_z = []
    for z_val in np.unique(z):
        idx = (z == z_val)
        y_idx = y[idx]
        y_pred_idx = y_pred[idx]
        tpr = TPR(y_idx, y_pred_idx, binary=binary)
        tpr_per_z.append(tpr)
    
    # take difference
    tpr_gap = tpr_per_z[1] - tpr_per_z[0]

    return tpr_gap


def TPR_gap_overall(y, y_pred, z, binary=True):
    """Calculate the True Positive Rate"""

    # calculate TPR gap per y
    if binary:
        # calculate TPR gap
        tpr_gap_overall = TPR_gap(y, y_pred, z, binary=True)
    else:

        # go over each category and calculate the TPR
        tpr_gaps = []
        for label in np.unique(y):

            # get the indices where y == label
            label_idx = (y == label)

            y_idx = y[label_idx]
            y_pred_idx = y_pred[label_idx]
            z_idx = z[label_idx]
           
            # calculate TPR gap
            tpr_gap_label = TPR_gap(y_idx, y_pred_idx, z_idx, binary=False)
            tpr_gaps.append(tpr_gap_label)
        
        # square the tpr gaps
        tpr_gaps = np.array(tpr_gaps)**2
        tpr_gap_overall = np.sqrt(np.mean(tpr_gaps))

    return tpr_gap_overall


def TPR_gap_multiclass(y, y_pred, z):

    # get all cases where y==y_pred
    correct = (y == y_pred)

    # get the unique values of y
    unique_y = np.unique(y)

    # calculate the TPR gap
    tpr_gaps = []
    for label in unique_y:

        # get the indices where y == label
        label_idx = (y == label)
        z_idx = z[label_idx]
        correct_idx = correct[label_idx]
        
        # get the indices where z == 1, z == 0
        idx_z_1 = (z_idx == 1)
        idx_z_0 = (z_idx == 0)

        # get the rate of correct predictions for z == 1 and z == 0
        tpr_z_1 = np.sum(correct_idx[idx_z_1]) / np.sum(idx_z_1)
        tpr_z_0 = np.sum(correct_idx[idx_z_0]) / np.sum(idx_z_0)

        # calculate the TPR gap, add
        tpr_label = tpr_z_1 - tpr_z_0
        tpr_gaps.append(tpr_label)

    
    # square the tpr gaps
    tpr_gaps = np.array(tpr_gaps)**2

    # take the mean of the tpr gaps
    tpr_gap_overall = np.sqrt(np.mean(tpr_gaps))

    return tpr_gap_overall


def calc_acc_per_class(y, y_pred):
    # calculate the accuracy for per class
    acc_per_class = {}
    for y_val in np.unique(y):
        idx = y == y_val
        y_idx = y[idx]
        y_pred_idx = y_pred[idx]
        acc = accuracy_score(y_idx, y_pred_idx)
        acc_per_class[y_val] = acc
    
    return acc_per_class


def calc_wg_acc(y, y_pred, g, return_acc_per_g=False):
    
    # calculate the accuracy for per group
    acc_per_g = {}
    for g_val in np.unique(g):
        idx = (g == g_val)
        y_idx = y[idx]
        y_pred_idx = y_pred[idx]
        acc = accuracy_score(y_idx, y_pred_idx)
        acc_per_g[g_val] = acc
    
    # take the worst group accuracy
    wg_acc = np.min(list(acc_per_g.values()))

    if return_acc_per_g:
        return wg_acc, acc_per_g
    return wg_acc


def evaluate_predictions(y, y_pred, z, task_name, g=None, y_pred_proba=None):
    """Evaluate predictions using accuracy, cross-entropy, and fairness metrics"""
    # Calculate accuracy - ensure we can handle multi-label y of (n, k)

    # convert y to class labels if needed
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
        binary = False
        acc_per_class = calc_acc_per_class(y, y_pred)
    else:
        binary = True
        acc_per_class = None

    # Calculate binary cross-entropy if probabilities are provided
    if y_pred_proba is not None:
        if binary:
            # For binary classification, ensure probabilities are for positive class
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba.flatten()
            cross_entropy = log_loss(y, y_pred_proba_pos, labels=[0, 1])
        else:
            # For multi-class classification
            cross_entropy = log_loss(y, y_pred_proba)
    else:
        cross_entropy = None

    # Ensure z is flattened to 1d
    if z.ndim > 1:
        z = z.flatten()
    
    # calculate the accuracy
    acc = accuracy_score(y, y_pred)
    
    # calculate the TPR rate - total, and per z
    if binary:
        tpr_gap = TPR_gap(y, y_pred, z, binary=binary)
    else:
        tpr_gap = TPR_gap_multiclass(y, y_pred, z)
    tpr = TPR(y, y_pred, binary=binary)

    # print the results
    if cross_entropy is not None:
        print(f"{task_name} Accuracy: {acc:.3f}, TPR: {tpr:.3f}, TPR Gap: {tpr_gap:.3f}, Cross-Entropy: {cross_entropy:.3f}")
    else:
        print(f"{task_name} Accuracy: {acc:.3f}, TPR: {tpr:.3f}, TPR Gap: {tpr_gap:.3f}")

    # calculate the worst group accuracy if g is not None
    if g is not None:
        wg_acc, acc_per_g = calc_wg_acc(y, y_pred, g, return_acc_per_g=True)
        print(f"Worst Group Accuracy: {wg_acc:.3f}")
        print("Accuracy per group:", {k: f"{v:.3f}" for k,v in acc_per_g.items()})
    else:
        wg_acc = None
        acc_per_g = None
    
    return acc, tpr, tpr_gap, wg_acc, acc_per_g, acc_per_class, cross_entropy


def save_results(args, results_dict, result_file_name='results.csv'):
    """Save results to CSV file"""
    results_dir = Path("results") / "last_layer"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / result_file_name
    
    # Define fieldnames for CSV
    fieldnames = results_dict[0].keys()
    print('fieldnames')
    
    # Create file with headers if it doesn't exist
    if (not csv_path.exists()):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Append results
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in results_dict:
            writer.writerow(row)
