import argparse
import numpy as np
from pathlib import Path
import h5py
from sklearn.linear_model import SGDClassifier, LogisticRegression  # Add this import
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from data import get_dataset_handler, BiosData, MultiLingualData, CelebAData
from proj import proj
import torch
from model import LastLayer, extract_bert_classifier, load_bert_model_tokenizer
import csv
import os
from utils import est_Cov, str_to_bool
import sys
from config import SAMPLE_PARAMS, LAST_LAYER_RETRAIN_PARAMS
from metrics import (
    calc_cov_sq_norm, TPR, TPR_gap, TPR_gap_overall, TPR_gap_multiclass,
    calc_acc_per_class, calc_wg_acc, evaluate_predictions, save_results
)


def main(args):
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split('-')]
    
    for seed in seeds:
        print(f"\nProcessing seed {seed}, method: {args.proj_method}")

        # get the handler based on the dataset
        data_handler = get_dataset_handler(args.dataset)
        
        # add a seed if args.add_seed=True
        if args.add_seed:
            args.model_name = args.model_name + f'_seed{seed}'
        
        # set the seed
        lambda_val = args.lambda_val if args.lambda_val is not None else 0
        
        # Create new classifier to be trained on projected data
        if lambda_val ==0:
            penalty = None
        else:
            penalty = 'l2'
        
        # Load data with embeddings
        print('Data from {}, with sample= {}'.format(args.dataset, args.sample_data))
        
        data = data_handler.prepare_data(
            load_test=True, 
            embeddings=True, 
            embedding_type=args.embedding_type,
            model_name= args.model_name,
            sample=args.sample_data,
            p_y_z=args.p_y_z,
            p_y=0.5,
            seed=seed,
        )
        
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        z_train, z_val, z_test = data['z_train'], data['z_val'], data['z_test']
        
        # the division of y, z per train/val/test is;
        print('p(y = 1) = {}, p(y =1  | z = 1) = {}, (train)'.format(np.mean(y_train), np.mean(y_train[z_train == 1])))
        print('p(y = 1) = {}, p(y =1  | z = 1) = {}, (val)'.format(np.mean(y_val), np.mean(y_val[z_val == 1])))
        print('p(y = 1) = {}, p(y =1  | z = 1) = {}, (test)'.format(np.mean(y_test), np.mean(y_test[z_test == 1])))
      
      
        d = X_train.shape[1]
     
        # determine the number of labels
        if args.sample_data:
            num_labels = len(np.unique(y_train))
        else:
            num_labels = data_handler.num_labels

        # Load original model only if not skipping
        original_model = None
        original_preds = None
        coef_orig = None
        intercept_orig = None
        
        if not args.skip_original:
            # Modify model path to include seed
            if args.add_seed:
                model_path = Path("models") / args.dataset / f"{args.model_name}"
            else:
                model_path = args.model_name
                
            # if model_name contains bert then load bert model
            if 'bert' in str(model_path).lower():
                # load the model
                model_path = Path("models") / args.dataset / f"{args.model_name}"
                print('model_path:', model_path)
                
                # load the model
                bert_model, _, _ = load_bert_model_tokenizer(
                    model_name=str(model_path),
                    num_labels=num_labels,
                    device='cpu',
                )
                original_model = extract_bert_classifier(bert_model)
                original_model = original_model.to(torch.float32)
                is_bert_model = True
                del bert_model
            else:
                # load the classifier_head.pt onto LastLayer
                state_dict = torch.load(model_path / 'classifier_head.pt', map_location='cpu')
                
                # change the keys if needed
                if 'linear' in state_dict or 'bias' in state_dict:
                    state_dict['linear.weight'] = state_dict.pop('weight')
                    state_dict['linear.bias'] = state_dict.pop('bias')
                
                # if the bias.dim() == 1, then num_labels = 1
                if state_dict['linear.bias'].shape[0] == 1:
                    num_labels_ = 1
                else:
                    num_labels_ = num_labels
                original_model =  LastLayer(d, num_labels_, coef=None, intercept=None)
                original_model.load_state_dict(state_dict)
        
            # Get predictions of the original model
            with torch.no_grad():
                # get the predictions of the original model
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

                # get the predictions of the original model
                y_train_pred = original_model.predict(X_train_tensor).cpu().numpy()
                y_val_pred = original_model.predict(X_val_tensor).cpu().numpy()
                y_test_pred = original_model.predict(X_test_tensor).cpu().numpy()
          
            # save the predictions
            original_preds = {'train': y_train_pred, 'val': y_val_pred, 'test': y_test_pred}

            # Save coefficients from original model for later use
            with torch.no_grad():
                coef_orig = original_model.linear.weight.cpu().numpy()
                intercept_orig = original_model.linear.bias.cpu().numpy()
                            
            # determine the shape of the coefficients
            if num_labels > 2:
                coef_orig = coef_orig.copy()
            else:
                # For binary classification, check shape and handle accordingly
                if coef_orig.shape[0] == 1:
                    coef_orig = coef_orig.copy()[0, :]  # Use first row if only one row exists
                else:
                    coef_orig = coef_orig.copy()[1, :]  # Use second row for standard binary case
                    
            # Handle intercept similarly
            if num_labels > 2:
                intercept_orig = intercept_orig.copy()
            else:
                # For binary classification, check shape and handle accordingly
                if len(intercept_orig) == 1:
                    intercept_orig = intercept_orig.copy()[0]  # Use first value if only one exists
                else:
                    intercept_orig = intercept_orig.copy()[1]  # Use second value for standard binary case

            acc_train = accuracy_score(y_train_classes, y_train_pred)
            acc_val = accuracy_score(y_val_classes, y_val_pred)
            acc_test = accuracy_score(y_test_classes, y_test_pred)
            print(f"Original Model Accuracy: Train: {acc_train:.3f}, Val: {acc_val:.3f}, Test: {acc_test:.3f}")

        # if num_labels > 1, then get the argmax of y_train
        if num_labels > 2:
            y_train_classes = np.argmax(y_train, axis=1)
            y_val_classes = np.argmax(y_val, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
        else:
            y_train_classes = y_train
            y_val_classes = y_val
            y_test_classes = y_test
       
        # Get projection method and fit once on training data
        proj_method = proj() if args.proj_method != 'orig' else None

        # standardize the data if specified
        if args.standardize:
            X_train_re, X_val_re, X_test_re = data_handler.standardize_X(X_train, X_val, X_test)
        else:
            X_train_re, X_val_re, X_test_re = X_train, X_val, X_test
        data_for_proj = {'X_train': X_train_re, 'X_val': X_val_re, 'X_test': X_test_re}

        # if the causal-LEACE variant is not none, then set the variant
        if args.proj_method == 'causal-LEACE':
            if args.causal_LEACE_variant == 'none':
                raise ValueError("Causal-LEACE variant must be specified")
            elif args.causal_LEACE_variant == 'oracle' or args.causal_LEACE_variant == 'estimate_y':
                proj_method.set_causal_LEACE_variant('use_y')
            else:
                proj_method.set_causal_LEACE_variant(args.causal_LEACE_variant)
        
        #  reshape y, z if needed
        if len(y_train.shape) == 1:
            y_train_proj = y_train.reshape(-1, 1)
        else:
            y_train_proj = y_train
        if len(z_train.shape) == 1:
            z_train_proj = z_train.reshape(-1, 1)
        else:
            z_train_proj = z_train
        
        
        # determine coef for opt-sep-proj if need be
        if args.proj_method == 'opt-sep-proj' and args.info_type == 'coef':
            coef = coef_orig.copy() if coef_orig is not None else None
            # reshape if need be
            if coef is not None and coef.ndim == 1:
                coef = coef.reshape(-1,1)
        elif args.proj_method == 'optimized-range':
            coef = coef_orig.copy() if coef_orig is not None else None
            # reshape if need be
            if coef is not None and coef.ndim == 1:
                coef = coef.reshape(-1,1)
        else:
            coef = None
        
            
        if (args.proj_method == 'LEACE'):
            proj_method.fit(X_train_re, z_train_proj, None, method='LEACE')
        elif (args.proj_method == 'opt-sep-proj'):
            proj_method.fit(X_train_re, z_train_proj, y_train_proj, method='opt-sep-proj', info_type=args.info_type, coef=coef)
        elif (args.proj_method == 'LEACE-no-whitening'):
            proj_method.fit(X_train_re, z_train_proj, None, method='LEACE-no-whitening')
        elif (args.proj_method == 'causal-LEACE'):
            proj_method.fit(X_train_re, z_train_proj, y_train_proj, method='causal-LEACE')
        elif (args.proj_method == 'SAL'):
            proj_method.fit(X_train_re, z_train_proj, None, method='SAL')
        elif (args.proj_method == 'orig'):
            pass
        elif (args.proj_method == 'balanced-LEACE'):

            if args.sample_data:
                
                

                # get the size of the smallest group
                if args.dataset =='multilingual':
                    get_group_v =  np.vectorize(data_handler.get_group_granular) 
                    z_train_balanced = (z_train.argmax(axis=1) + 1).reshape(-1, 1)
                else:
                    get_group_v = np.vectorize(data_handler.get_group)
                    z_train_balanced = z_train
                g_train = get_group_v(y_train.squeeze(), z_train_balanced.squeeze())
                n_smallest_group =np.bincount(g_train)[1:].min()
                n_balanced = n_smallest_group * len(np.unique(g_train))
                
                if args.dataset == 'multilingual' and len(np.unique(z_train)) > 2:
                    indices_balanced = data_handler.sample_by_probabilities_granular(
                        y_train,
                        z_train_balanced,
                        p_y_z=args.p_y_z,
                        p_y=args.p_y,
                        n=n_balanced,
                        seed=SAMPLE_PARAMS['sample_seed']
                    )
                else:
                    indices_balanced = data_handler.sample_by_probabilities(
                        y_train,
                        z_train_balanced,
                        p_y_z=0.5,
                        p_y=0.5,
                        n=n_balanced,
                        seed=SAMPLE_PARAMS['sample_seed']
                    )

               
                y_train_balanced, z_train_balanced, X_train_balanced = y_train[indices_balanced], z_train[indices_balanced], X_train[indices_balanced]
                proj_method.fit(X_train_balanced, z_train_balanced, None, method='LEACE')
            else:
                raise ValueError("Balanced LEACE requires sampling data")
        elif (args.proj_method == 'optimized-range'):
            
            proj_method.fit(X_train_re, z_train_proj, y_train_proj, method='optimized-range', coef=coef, w=None)
        elif (args.proj_method == 'optimized-range-reweighted'):
            
            # Determine weights for reweighting
            # Calculate weights for balanced groups
            # Get group assignments using vectorized function
            if args.dataset == 'multilingual':
                get_group_v = np.vectorize(data_handler.get_group_granular)
                z_for_groups = z_train_proj.argmax(axis=1) + 1 if z_train_proj.ndim > 1 else z_train_proj.flatten()
            else:
                get_group_v = np.vectorize(data_handler.get_group)
                z_for_groups = z_train_proj.flatten()
            
            # Get y for group calculation
            if y_train_proj.ndim > 1 and y_train_proj.shape[1] > 1:
                # Multi-class case
                y_for_groups = y_train_proj.argmax(axis=1)
            else:
                y_for_groups = y_train_proj.flatten()
            
            # Calculate group assignments
            groups = get_group_v(y_for_groups, z_for_groups)
            unique_groups, group_counts = np.unique(groups, return_counts=True)
            
            print(f"Group counts before reweighting: {dict(zip(unique_groups, group_counts))}")
            
            # Calculate weights for balanced groups
            n = len(groups)  # Total number of samples
            w = np.ones((n, 1))
            p_g = 1 / len(unique_groups)  # Target probability for each group
            for group_id, group_count in zip(unique_groups, group_counts):
                # n per group
                p_g_tr = group_count / n 
                w_g = p_g / p_g_tr  # Weight for this group
                w[groups == group_id] = w_g
                print(f" Group {group_id} weight: {w_g:.4f} p_g_tr: {p_g_tr:.4f} p_g: {p_g:.4f}")
          
            # Normalize weights to sum to n
            w = (w / np.sum(w)) *n 
            
            print('Weights for groups:', {group: w[groups == group].mean() for group in unique_groups})
            print('Avg. weight:', np.mean(w))
            
            proj_method.fit(X_train_re, z_train_proj, y_train_proj, method='optimized-range', coef=coef, w=w)
        else:
            raise ValueError(f"Unknown projection method: {args.proj_method}")
        
        # Remove the manual proj_name logic since it's now handled by the method choice
        proj_name = args.proj_method
        
        # Apply fitted projection to all splits
        projected_embeddings = {}
        splits_map = {'train': 'train', 'val': 'val', 'test': 'test'}
        
        # Apply projection to all splits
        for split, data_split in splits_map.items():
            X = data_for_proj[f'X_{data_split}']

            if args.proj_method == 'orig':
                # if the projection method is orig, then do not apply any projection
                projected_embeddings[split] = X
            elif args.proj_method == 'causal-LEACE':
                # if the causal-LEACE variant is oracle, then apply the projection with knowledge of y
                if  args.causal_LEACE_variant == 'oracle':
                    y = data[f'y_{data_split}'].reshape(-1, 1)
                elif args.causal_LEACE_variant == 'estimate_y':
                    if original_preds is not None:
                        y = original_preds[split].reshape(-1, 1)
                    else:
                        y = None
                elif args.causal_LEACE_variant == 'range' or args.causal_LEACE_variant == 'balance':
                    y = None
                projected_embeddings[split] = proj_method.apply_projection(X, y)
            else:
                projected_embeddings[split] = proj_method.apply_projection(X)
                print(f"Projected {split} embeddings with shape: {projected_embeddings[split].shape}")
                
        # Check covariance properties on training set
        cov_x_y = est_Cov(X_train, y_train)
        cov_x_z = est_Cov(X_train, z_train)

        # after the projection, determine Cov(x, y) and Cov(x, z)
        cov_x_y_proj = est_Cov(projected_embeddings['train'], y_train)
        cov_x_z_proj = est_Cov(projected_embeddings['train'], z_train)

        # measure the sq. norm of each of the covariance matrices, after projection
        sq_norm_cov_x_y =calc_cov_sq_norm(cov_x_y)
        sq_norm_cov_x_z = calc_cov_sq_norm(cov_x_z)
        sq_norm_cov_x_y_proj = calc_cov_sq_norm(cov_x_y_proj)
        sq_norm_cov_x_z_proj = calc_cov_sq_norm(cov_x_z_proj)
        perc_sq_norm_cov_x_y_proj = sq_norm_cov_x_y_proj/sq_norm_cov_x_y
        perc_sq_norm_cov_x_z_proj = sq_norm_cov_x_z_proj/sq_norm_cov_x_z
        
        # if the dataset is Waterbirds, then set class_weight to True
        if args.dataset=='Waterbirds':
            class_weight = 'balanced'
            print("Using class_weight=True for Waterbirds dataset")
        else:
            class_weight = None

        # Create new classifier to be trained on projected data
        new_classifier =SGDClassifier(loss='log_loss', 
                                        alpha=lambda_val,
                                        learning_rate='optimal',
                                        penalty=penalty,
                                        l1_ratio=0.0,
                                        warm_start=True,
                                        random_state=seed,
                                        class_weight=class_weight,
                                        tol=1e-3)
        # Initialize classifier with original coefficients if available
        if coef_orig is not None and intercept_orig is not None:
            new_classifier.coef_ = coef_orig.copy()
            new_classifier.intercept_ = intercept_orig.copy()
        
       
        new_classifier.fit(projected_embeddings['train'], y_train_classes)

        # Initialize results list
        results = []
        
        # Evaluate on all splits
        print("\nEvaluation Results:")
        print("-" * 50)

        splits = ['train', 'val', 'test']
        for split in splits:
            print(f"\n{split.upper()} Split:")
            X_proj = projected_embeddings[split]
            y = data[f'y_{split}']
            z = data[f'z_{split}']
            
            
            # check shape
            if len(y.shape) > 1:
                if y.shape[1] == 1:
                    y = y.ravel()
            if len(z.shape) > 1:
                if z.shape[1] == 1:
                    z = z.ravel()

            # evaluate_predictions(y, y_pred, y_pred_proba, z, task_name):
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

            # Original BERT last layer on projected data - only if not skipping
            if not args.skip_original:
                print("\nBERT Last Layer (no retraining):")
                
                # get pred from original model
                with torch.no_grad():
                    X_proj_tensor = torch.tensor(X_proj, dtype=torch.float32)
                    y_pred_orig = original_model.predict(X_proj_tensor).cpu().numpy()
                    # Get probabilities for cross-entropy calculation
                    y_pred_proba_orig = original_model.predict_proba(X_proj_tensor).cpu().numpy()
              
                acc_orig, tpr_orig, tpr_gap_orig, wg_acc_orig, acc_per_g_orig, acc_per_class_orig, ce_orig = evaluate_predictions(y, y_pred_orig, z, "Main Task", g, y_pred_proba_orig)

                 # Save original classifier results with cross-entropy
                result = {
                    'dataset': args.dataset,
                    'model_name': args.model_name,
                    'embedding_type': args.embedding_type,
                    'proj_method': proj_name,
                    'split': split,
                    'classifier_type': 'original',
                    'accuracy': acc_orig,
                    'tpr': tpr_orig,
                    'tpr_gap': tpr_gap_orig,
                    'worst_group_accuracy': wg_acc_orig,
                    'cross_entropy': ce_orig,
                }

                if args.sample_data:
                    # save p_y, p_y_z
                    result['p_y'] = 0.5
                    result['p_y_z'] = args.p_y_z

                # Add accuracy per group if available
                if acc_per_g_orig is not None:
                    for g_val, acc in acc_per_g_orig.items():
                        result[f'acc_g{g_val}'] = acc
                
                # Add accuracy per class if available
                if acc_per_class_orig is not None:
                    for c_val, acc in acc_per_class_orig.items():
                        result[f'acc_c{c_val}'] = acc

                results.append(result)

            # Retrained classifier on projected data - only run if not only_original
            if not args.only_original:
                print("\nRetrained Classifier on Projected Data:")
                y_pred_new = new_classifier.predict(X_proj)
                # Get probabilities for cross-entropy calculation
                y_pred_proba_new = new_classifier.predict_proba(X_proj)
                acc_new, tpr_new, tpr_gap_new, wg_acc_new, acc_per_g_new, acc_per_class_new, ce_new = evaluate_predictions(y, y_pred_new, z, "Main Task", g, y_pred_proba_new)
                
                # Save retrained classifier results with cross-entropy
                result = {
                    'dataset': args.dataset,
                    'model_name': args.model_name,
                    'embedding_type': args.embedding_type,
                    'proj_method': proj_name,
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

        
        # add several results that apply to all splits - Cov(x, y), Cov(x, z), Cov(x, y)_proj, Cov(x, z)_proj, % Cov(x, y)_proj, % Cov(x, z)_proj
        for result in results:
            result['seed'] = seed
            result['lambda_val'] = args.lambda_val
            result['sq_norm_cov_x_y'] = sq_norm_cov_x_y
            result['sq_norm_cov_x_z'] = sq_norm_cov_x_z
            result['sq_norm_cov_x_y_proj'] = sq_norm_cov_x_y_proj
            result['sq_norm_cov_x_z_proj'] = sq_norm_cov_x_z_proj
            result['perc_sq_norm_cov_x_y_proj'] = perc_sq_norm_cov_x_y_proj
            result['perc_sq_norm_cov_x_z_proj'] = perc_sq_norm_cov_x_z_proj


            # if the projection method is causal-LEACE, then add the variant
            result['causal_LEACE_variant'] = args.causal_LEACE_variant


        print('|| Cov(x, y) ||:', sq_norm_cov_x_y)
        print('|| Cov(x, z) ||:', sq_norm_cov_x_z)
        print('|| Cov(x, y)_proj ||:', sq_norm_cov_x_y_proj)
        print('|| Cov(x, z)_proj ||:', sq_norm_cov_x_z_proj)
        print('% ||Cov(x, y)_proj||:', perc_sq_norm_cov_x_y_proj)
        print('% ||Cov(x, z)_proj||:', perc_sq_norm_cov_x_z_proj)
        



        
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
                f.write(f"Projection Method: {args.proj_method}\n")
                f.write(f"Sample Data: {args.sample_data}\n")
                f.write(f"p_y_z: {args.p_y_z}\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Embedding Type: {args.embedding_type}\n")
                f.write(f"penalty: {lambda_val}\n")
                f.write(f'causal_LEACE_variant: {args.causal_LEACE_variant}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset to evaluate (e.g., bios)")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Model name or path")
    parser.add_argument("--embedding_type", type=str, default="cls",
                      choices=['cls', 'pooler', 'mean', 'last' ,'raw'],
                      help="Type of embedding to use")
    parser.add_argument("--proj_method", type=str, default="LEACE",
                      choices=['LEACE', 'opt-sep-proj', 'orig', 'LEACE-no-whitening', 'causal-LEACE', 'SAL', 'balanced-LEACE', 'optimized-range', 'optimized-range-reweighted'],
                      help="Projection method to use")
    parser.add_argument("--causal_LEACE_variant", type=str, default='none',
                        choices=['oracle', 'estimate_y', 'naive', 'range', 'none', 'balance'])
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
                        help='Standardize data before applying projection')
    parser.add_argument("--lambda_val", type=float, default=None,
                      help="Penalty strength (lambda) for the SGD classifier")
    parser.add_argument('--add_seed', type=str, default='False',
                        help='Add seed to model name')
    parser.add_argument('--info_type', type=str, default='Cov',
                        choices=['Cov', 'coef'],
                        help='Type of information to use for opt-sep-proj projection')
    parser.add_argument('--only_original', type=str, default='False',
                        help='Only evaluate original classifier post-projection, skip retraining')
    parser.add_argument('--skip_original', type=str, default='False',
                        help='Skip loading and evaluating original model, only retrain')
                        
    args = parser.parse_args()
    args.sample_data = str_to_bool(args.sample_data)
    args.save = str_to_bool(args.save)
    args.standardize = str_to_bool(args.standardize)
    args.add_seed = str_to_bool(args.add_seed)
    args.only_original = str_to_bool(args.only_original)
    args.skip_original = str_to_bool(args.skip_original)
    main(args)
