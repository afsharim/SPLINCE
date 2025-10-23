import argparse
import numpy as np
from pathlib import Path
import h5py
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from data import get_dataset_handler, BiosData, MultiLingualData, CelebAData, WaterbirdsData
import torch
from model import GDROModel
import csv
import os
from utils import str_to_bool
import sys
from config import SAMPLE_PARAMS, LAST_LAYER_RETRAIN_PARAMS
from metrics import (
    evaluate_predictions, save_results
)


def SUBG(X, y, g, seed=42):
    """
    Subgroup sampling: get equal number of samples per group
    
    Args:
        X: input features
        y: labels  
        g: group assignments
        seed: random seed for sampling
        
    Returns:
        X_sampled, y_sampled: subsampled data with equal representation per group
    """
    np.random.seed(seed)
    
    unique_groups, group_counts = np.unique(g, return_counts=True)
    print(f"Original group counts: {dict(zip(unique_groups, group_counts))}")
    
    # Get size of smallest group
    n_smallest_group = group_counts.min()
    print(f"Sampling {n_smallest_group} samples per group")
    
    # Sample equal number from each group
    sampled_indices = []
    for group_id in unique_groups:
        group_indices = np.where(g == group_id)[0]
        sampled_group_indices = np.random.choice(
            group_indices, 
            size=n_smallest_group, 
            replace=False
        )
        sampled_indices.extend(sampled_group_indices)
    
    sampled_indices = np.array(sampled_indices)
    
    return X[sampled_indices], y[sampled_indices]


def main(args):
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split('-')]
    
    for seed in seeds:
        print(f"\nProcessing seed {seed}, method: {args.method}")

        # get the handler based on the dataset
        data_handler = get_dataset_handler(args.dataset)
        
        # add a seed if args.add_seed=True
        if args.add_seed:
            args.model_name = args.model_name + f'_seed{seed}'
        
        # set the seed
        lambda_val = args.lambda_val if args.lambda_val is not None else 0
        
        # Create new classifier to be trained on original data
        if lambda_val == 0:
            penalty = None
        else:
            penalty = 'l2'
        
        # Load data with embeddings
        print('Data from {}, with sample= {}'.format(args.dataset, args.sample_data))
        
        data = data_handler.prepare_data(
            load_test=True, 
            embeddings=True, 
            embedding_type=args.embedding_type,
            model_name=args.model_name,
            sample=args.sample_data,
            p_y_z=args.p_y_z,
            p_y=0.5,
            seed=seed,
        )
        
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        z_train, z_val, z_test = data['z_train'], data['z_val'], data['z_test']
        
        d = X_train.shape[1]
     
        # determine the number of labels
        if args.sample_data:
            num_labels = len(np.unique(y_train))
        else:
            num_labels = data_handler.num_labels

        
        # if num_labels > 1, then get the argmax of y_train
        if num_labels > 2:
            y_train_classes = np.argmax(y_train, axis=1)
            y_val_classes = np.argmax(y_val, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
        else:
            y_train_classes = y_train
            y_val_classes = y_val
            y_test_classes = y_test

        # standardize the data if specified
        if args.standardize:
            X_train_re, X_val_re, X_test_re = data_handler.standardize_X(X_train, X_val, X_test)
        else:
            X_train_re, X_val_re, X_test_re = X_train, X_val, X_test

        # Apply method-specific preprocessing
        if args.method == 'SUBG':
            # Subgroup sampling on training set
            print("Applying subgroup sampling on training set...")
            
            # Get groups for training set
            if args.dataset == 'multilingual':
                get_group_v = np.vectorize(data_handler.get_group_granular)
                z_for_groups = z_train.argmax(axis=1) + 1 if z_train.ndim > 1 else z_train.flatten()
            else:
                get_group_v = np.vectorize(data_handler.get_group)
                z_for_groups = z_train.flatten()
            
            # Get y for group calculation
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_for_groups = y_train.argmax(axis=1)
            else:
                y_for_groups = y_train.flatten()
            
            # Calculate group assignments
            groups = get_group_v(y_for_groups, z_for_groups)
            
            # Apply SUBG to training data
            X_train_re, y_train_classes = SUBG(X_train_re, y_train_classes, groups, seed=seed)
            print(f"Training data shape after subgroup sampling: {X_train_re.shape}")
        
        elif args.method == 'GDRO':
            # GDRO: Group Distributionally Robust Optimization
            print("Applying Group Distributionally Robust Optimization (GDRO)...")
            
            # Get groups for training set
            if args.dataset == 'multilingual':
                get_group_v = np.vectorize(data_handler.get_group_granular)
                z_for_groups = z_train.argmax(axis=1) + 1 if z_train.ndim > 1 else z_train.flatten()
            else:
                get_group_v = np.vectorize(data_handler.get_group)
                z_for_groups = z_train.flatten()
            
            # Get y for group calculation
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_for_groups = y_train.argmax(axis=1)
            else:
                y_for_groups = y_train.flatten()
            
            # Calculate group assignments
            groups_train = get_group_v(y_for_groups, z_for_groups)
            
            # Get groups for validation set for GDRO optimization
            if args.dataset == 'multilingual':
                z_val_for_groups = z_val.argmax(axis=1) + 1 if z_val.ndim > 1 else z_val.flatten()
            else:
                z_val_for_groups = z_val.flatten()
            
            if y_val.ndim > 1 and y_val.shape[1] > 1:
                y_val_for_groups = y_val.argmax(axis=1)
            else:
                y_val_for_groups = y_val.flatten()
            
            groups_val = get_group_v(y_val_for_groups, z_val_for_groups)
            
            print(f"Training group counts: {dict(zip(*np.unique(groups_train, return_counts=True)))}")
            print(f"Validation group counts: {dict(zip(*np.unique(groups_val, return_counts=True)))}")
            
            # Create base SGD classifier for GDRO
            base_classifier = SGDClassifier(
                loss='log_loss',
                alpha=lambda_val,
                learning_rate='optimal',
                penalty=penalty,
                l1_ratio=0,
                warm_start=True,
                random_state=seed,
                tol=1e-3
            )
            
            # Create GDRO model wrapper
            gdro_model = GDROModel(base_classifier)
            
            # GDRO hyperparameters (can be made configurable)
            T = args.gdro_epochs  # number of epochs
            batch_size = args.gdro_batch_size
            eta_param = args.gdro_lr_param  # learning rate for parameters
            eta_q = args.gdro_lr_q  # learning rate for group weights
            C = args.gdro_C  # regularization parameter
            
            # Run GDRO optimization
            best_Beta, best_worst_loss, best_epoch = gdro_model.optimize_GDRO_via_SGD(
                X_train_re, y_train_classes.reshape(-1, 1), groups_train,
                X_val_re, y_val_classes.reshape(-1, 1), groups_val,
                T=T, batch_size=batch_size, eta_param=eta_param, eta_q=eta_q, C=C,
                early_stopping=args.gdro_early_stopping, patience=args.gdro_patience
            )
            
            # Update the classifier to use the best parameters
            new_classifier = gdro_model.base_model
            print(f"GDRO optimization completed. Best epoch: {best_epoch}, Best worst group loss: {best_worst_loss:.4f}")
            
        elif args.method == 'orig':
            # ERM: use original data as-is
            print("Using ERM (original data)")
            
            # Create new classifier to be trained on original data
            new_classifier = SGDClassifier(loss='log_loss', 
                                            alpha=lambda_val,
                                            learning_rate='optimal',
                                            penalty=penalty,
                                            l1_ratio=0,
                                            warm_start=True,
                                            random_state=seed,
                                            tol=1e-3)
            
            new_classifier.fit(X_train_re, y_train_classes)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        # For non-GDRO methods, train the classifier if not already done
        if args.method != 'GDRO' and args.method != 'orig':
            # Create new classifier to be trained on (potentially subsampled) data
            new_classifier = SGDClassifier(loss='log_loss', 
                                            alpha=lambda_val,
                                            learning_rate='optimal',
                                            penalty=penalty,
                                            l1_ratio=0,
                                            warm_start=True,
                                            random_state=seed,
                                            tol=1e-3)
            
            new_classifier.fit(X_train_re, y_train_classes)

        # Initialize results list
        results = []
        
        # Evaluate on all splits
        print("\nEvaluation Results:")
        print("-" * 50)

        splits = ['train', 'val', 'test']
        X_splits = {'train': X_train_re if args.method == 'orig' else data_handler.standardize_X(data['X_train'], data['X_val'], data['X_test'])[0] if args.standardize else data['X_train'], 
                   'val': X_val_re, 'test': X_test_re}
        
        for split in splits:
            print(f"\n{split.upper()} Split:")
            X = X_splits[split]
            y = data[f'y_{split}']
            z = data[f'z_{split}']
            print(
                f"Shape of X: {X.shape}, y: {y.shape}, z: {z.shape}"
            )
            
            # check shape
            if len(y.shape) > 1:
                if y.shape[1] == 1:
                    y = y.ravel()
            if len(z.shape) > 1:
                if z.shape[1] == 1:
                    z = z.ravel()

            if args.sample_data:
                # if the z is one-hot encoded, then turn to 1d via argmax
                if z.ndim > 1:
                    z = np.argmax(z, axis=1) + 1
                if args.dataset == 'bios':
                    # get group for bios
                    get_group_v = np.vectorize(data_handler.get_group)
                elif args.dataset == 'Waterbirds':
                    # get group for waterbirds
                    get_group_v = np.vectorize(data_handler.get_group)
                elif args.dataset == 'CelebA':
                    # get group for CelebA
                    get_group_v = np.vectorize(data_handler.get_group)
                elif args.dataset == 'multilingual':
                    # get group for multilingual
                    get_group_v = np.vectorize(data_handler.get_group_granular)
                    
                g = get_group_v(y, z)
            else:
                g = None

            # Retrained classifier evaluation
            if not args.only_original:
                print(f"\nRetrained Classifier ({args.method}):")
                y_pred_new = new_classifier.predict(X)
                # Get probabilities for cross-entropy calculation
                y_pred_proba_new = new_classifier.predict_proba(X)
                print('Shape of y_pred_proba_new:', y_pred_proba_new.shape)
                print('Shape of y_pred_new:', y_pred_new.shape)
                print('shape of y:', y.shape)
                acc_new, tpr_new, tpr_gap_new, wg_acc_new, acc_per_g_new, acc_per_class_new, ce_new = evaluate_predictions(y, y_pred_new, z, "Main Task", g, y_pred_proba_new)
                
                # Save retrained classifier results with cross-entropy
                result = {
                    'dataset': args.dataset,
                    'model_name': args.model_name,
                    'embedding_type': args.embedding_type,
                    'method': args.method,
                    'split': split,
                    'classifier_type': 'retrained',
                    'accuracy': acc_new,
                    'tpr': tpr_new,
                    'tpr_gap': tpr_gap_new,
                    'worst_group_accuracy': wg_acc_new,
                    'cross_entropy': ce_new,
                }

                if args.sample_data:
                    # save p_y, p_y_z
                    result['p_y'] = 0.5
                    result['p_y_z'] = args.p_y_z
                    
                # Add accuracy per group if available
                if acc_per_g_new is not None:
                    for g_val, acc in acc_per_g_new.items():
                        result[f'acc_g{g_val}'] = acc
                
                # Add accuracy per class if available
                if acc_per_class_new is not None:
                    for c_val, acc in acc_per_class_new.items():
                        result[f'acc_c{c_val}'] = acc

                results.append(result)

        # add seed and lambda_val to all results
        for result in results:
            result['seed'] = seed
            result['lambda_val'] = args.lambda_val

        # Save results for this seed
        if args.save:
            save_results(args, results, args.result_file_name)

            # save several parameters to a file
            results_dir = Path("results") / "last_layer"
            results_dir.mkdir(parents=True, exist_ok=True)
            params_file = results_dir / f"params_{args.result_file_name}"
            with open(params_file, 'a') as f:
                f.write(f"Seed: {seed}\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Method: {args.method}\n")
                f.write(f"Sample Data: {args.sample_data}\n")
                f.write(f"p_y_z: {args.p_y_z}\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Embedding Type: {args.embedding_type}\n")
                f.write(f"penalty: {lambda_val}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset to evaluate (e.g., bios)")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Model name or path")
    parser.add_argument("--embedding_type", type=str, default="cls",
                      choices=['cls', 'pooler', 'mean', 'last', 'raw'],
                      help="Type of embedding to use")
    parser.add_argument("--method", type=str, default="orig",
                      choices=['orig', 'SUBG', 'GDRO'],
                      help="Method to use: 'orig' for ERM, 'SUBG' for subgroup sampling, 'GDRO' for group distributionally robust optimization")
    
    # GDRO-specific arguments
    parser.add_argument("--gdro_epochs", type=int, default=20,
                      help="Number of epochs for GDRO optimization")
    parser.add_argument("--gdro_batch_size", type=int, default=128,
                      help="Batch size for GDRO optimization")
    parser.add_argument("--gdro_lr_param", type=float, default=0.01,
                      help="Learning rate for model parameters in GDRO")
    parser.add_argument("--gdro_lr_q", type=float, default=0.1,
                      help="Learning rate for group weights in GDRO")
    parser.add_argument("--gdro_C", type=float, default=0.0,
                      help="Regularization parameter for GDRO")
    parser.add_argument("--gdro_early_stopping", type=str, default='True',
                      help="Whether to use early stopping in GDRO")
    parser.add_argument("--gdro_patience", type=int, default=3,
                      help="Patience for early stopping in GDRO")
    
    parser.add_argument("--device", type=str,
                      default="cpu",
                      help="Device to use for computation")
    parser.add_argument("--seeds", type=str, default="1",
                      help="Seeds to evaluate, separated by hyphens (e.g., '1-2-3')")
    parser.add_argument("--sample_data", type=str,
                        default='False',
                        help="Sample data to balance classes")
    parser.add_argument("--p_y_z", type=float, default=0.5,
                      help="P(Y=1|Z=1) for sampled data")
    parser.add_argument("--save", type=str, default='False',
                        help="Save results to CSV")
    parser.add_argument("--result_file_name", type=str, default='results.csv',
                        help="Name of results file to save")
    parser.add_argument('--standardize', type=str, default='False',
                        help='Standardize data before training')
    parser.add_argument("--lambda_val", type=float, default=None,
                      help="Penalty strength (lambda) for the SGD classifier")
    parser.add_argument('--add_seed', type=str, default='False',
                        help='Add seed to model name')
    parser.add_argument('--only_original', type=str, default='False',
                        help='Only evaluate original classifier, skip retraining')
                        
    args = parser.parse_args()
    args.sample_data = str_to_bool(args.sample_data)
    args.save = str_to_bool(args.save)
    args.standardize = str_to_bool(args.standardize)
    args.add_seed = str_to_bool(args.add_seed)
    args.only_original = str_to_bool(args.only_original)
    args.gdro_early_stopping = str_to_bool(args.gdro_early_stopping)
    main(args)
