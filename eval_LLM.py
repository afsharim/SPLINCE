import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import map_model_name
import argparse
import os
from utils import str_to_bool
from sklearn.linear_model import LinearRegression
import pickle
from utils import one_sided_ttest, get_se_coef
from model import load_model_and_tokenizer
import torch

def test_coefficient_difference(method1, method2, coef_idx, X_test, alternative='greater'):
    """
    Test if coefficient of method1 is significantly different from method2.
    
    Args:
        method1: Dict containing 'model' and 'y' for first method
        method2: Dict containing 'model' and 'y' for second method
        coef_idx: Index of coefficient to test (0=stereotype, 1=fact)
        X_test: Feature matrix for testing
        alternative: 'greater' or 'less' for one-sided test
    
    Returns:
        Dict with test statistic, p-value, and coefficient values
    """
    # Get coefficients
    coef1 = method1['model'].coef_[coef_idx]
    coef2 = method2['model'].coef_[coef_idx]
    
    # Calculate standard error for method1
    se1 = get_se_coef(method1['model'], X_test, method1['y'])[coef_idx]
    
    # Perform one-sided t-test
    test_result = one_sided_ttest(
        coef1, 
        se1, 
        null_hypothesis=coef2, 
        alternative=alternative
    )
    
    return {
        'coef1': coef1,
        'coef2': coef2,
        't_stat': test_result['t_stat'],
        'p_value': test_result['p_value']
    }

def evaluate_gender_scores(df):
    """Run regression analysis on gender scores"""

   
    # Prepare data for regression
    X = df[['stereotype_score', 'fact_score']]
    y = df['gender_score']

    # calculate the percentage of cases where gender_score is not 0
    non_zero = np.count_nonzero(y)
    total = len(y)

    # Fit regression model
    model = LinearRegression().fit(X, y)
    
    # get the coef for the stereotype_score and fact_score
    coef = model.coef_
 
    return model
def load_data(args, method=None):
    """
    Load data based on provided arguments.
    
    Args:
        args: Command line arguments
        method: Specific projection method to load (overrides args.projection_method)
        subset: Whether to load subset of test data
    
    Returns:
        DataFrame with loaded data
    """
    # Map model name to short version
    model_name_short = map_model_name(args.model_name)
    
    # Use provided method if specified, otherwise use from args
    projection_method = method if method else args.projection_method
    
    # Construct file suffix if projection was applied
    suffix = ""
    if args.apply_projection and projection_method != "orig":
        if args.layers == "lm_head":
            suffix = f"_{projection_method}_lm_head_{args.embedding_strategy}"
        elif args.layers == "all":
            suffix = f"_{projection_method}_all_{args.embedding_strategy}"
        else:
            layers_str = "_".join([str(layer) for layer in args.layers.split(",")])
            suffix = f"_{projection_method}_{layers_str}__{args.embedding_strategy}"
    
    # Construct file path
    file_path = f'data/{args.dataset}_professions_{model_name_short}{suffix}.csv'
    print('loading file:', file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found at {file_path}. Make sure to run calc_pred_LLM.py first.")

    # Load data
    df = pd.read_csv(file_path)
    
    # Add method column for identification
    df['projection_method'] = projection_method
    df['embedding_strategy'] = args.embedding_strategy

  
    
    return df

def main(args):
    # If multiple methods specified, evaluate each one
    all_methods = args.projection_method.split(',')
    results = []
    
    for method in all_methods:
        print(f"\nEvaluating projection method: {method}")
        # Load data for this method
        try:
            df = load_data(args, method=method.strip())

            # if split==test, set df to 
            if args.split == 'test':
                df = df[df['split'] == 'test']
            # if split==train, set df to
            elif args.split == 'train':
                df = df[df['split'] == 'train']
            # select subset of test data without correlation
            elif args.split == 'test_no_cor':
   
                # load the subset
                filename = f'data/{args.dataset}/test_subset_indices.pkl'
                with open(filename, 'rb') as f:
                    test_indices = pickle.load(f)
                
                df_test = df[df['split'] == 'test']
                df = df_test.iloc[test_indices]

            
            # Evaluate gender scores
            if args.eval_type == 'gender_score':
                model = evaluate_gender_scores(df)
                print('method is:', method)

                # save the model if method is opt_sep, leace
                if method == 'opt-sep-proj':
                    model_opt_sep = model
                    y_opt_sep = df['gender_score']
                elif method == 'LEACE':
                    model_leace = model
                    y_leace = df['gender_score']
                
                # Store results
                result = {
                    'method': method,
                    'embedding_strategy': args.embedding_strategy,
                    'stereotype_coef': model.coef_[0],
                    'fact_coef': model.coef_[1],
                    'intercept': model.intercept_,
                    'avg_gender_score': df['gender_score'].mean()
                }
                results.append(result)
                
                 # if model_opt_sep and model_leace are defined, test if the coefficients are significantly different
                if 'model_opt_sep' in locals() and 'model_leace' in locals():
                    
                    # Prepare data for testing
                    X_test = df[['stereotype_score', 'fact_score']]
                    print('n in X_test:', len(X_test))
                    method_opt_sep = {'model': model_opt_sep, 'y': y_opt_sep}
                    method_leace = {'model': model_leace, 'y': y_leace}

                    # Test fact coefficient
                    fact_test = test_coefficient_difference(
                        method_opt_sep, method_leace, coef_idx=1,
                        X_test=X_test, alternative='greater'
                    )

                    # Test stereotype coefficient
                    stereotype_test = test_coefficient_difference(
                        method_opt_sep, method_leace, coef_idx=0,
                        X_test=X_test, alternative='less'
                    )
                    print("\nCoefficient Difference Test:")
                    print("-" * 50)
                    print(f"Fact Coefficient: {fact_test}")
                    print(f"Stereotype Coefficient: {stereotype_test}")
                
            # if the eval_type is accuracy
            elif args.eval_type == 'prediction_winobias':
                
                # get the correct otkens
                correct_tokens = df['correct_tok'].values
                
                # get the predicted tokens
                predicted_tokens = df['predicted_tokens'].values
                
                # get correct variable
                correct = correct_tokens == predicted_tokens
                df['correct'] = correct
                
                # show the first 5 rows from correct_tok and predicted_tokens
                print("First 5 rows of correct and predicted tokens:")
                print("-" * 50)
                print(df[['correct_tok', 'predicted_tokens']].head())
                
                # get the accuracy
                accuracy = np.mean(correct)
                print(f"Accuracy: {accuracy:.4f}")
                
                # based on pronoun, map gender
                he_she_map = {
                    'he':'m', 'him':'m', 'his':'m',
                    'she':'f', 'her':'f', 'hers':'f'
                }
                df['gender'] = df['pronoun'].map(he_she_map)
                
                # split the accuracy per type_text, anti_stereotype, pronoun
                df_accuracy = df.groupby(['type_text', 'anti_stereotype', 'gender']).agg(
                    accuracy=('correct', 'mean')
                ).reset_index()
                print("\nAccuracy by Type Text, Anti Stereotype, Profession, Pronoun:")
                print("-" * 50)
                print(df_accuracy.to_string(index=False))
                
                # calculate difference: anti-stereotype - stereotype accuracy
                anti_stereotype_correct = df[df['anti_stereotype'] == 1]['correct'].values
                stereotype_correct = df[df['anti_stereotype'] == 0]['correct'].values
                avg_acc_anti_stereotype = np.mean(anti_stereotype_correct)
                avg_acc_stereotype = np.mean(stereotype_correct)
                acc_diff = avg_acc_stereotype - avg_acc_anti_stereotype
                print(f"\nAccuracy Difference (Anti-Stereotype - Stereotype): {acc_diff}")
                print(f"Average Anti-Stereotype Accuracy: {avg_acc_anti_stereotype:.4f}")
                print(f"Average Stereotype Accuracy: {avg_acc_stereotype:.4f}")
                
                # calculate difference: male - female accuracy
                male_correct = df[df['gender'] == 'm']['correct'].values
                female_correct = df[df['gender'] == 'f']['correct'].values
                avg_acc_male = np.mean(male_correct)
                avg_acc_female= np.mean(female_correct)
                acc_diff = avg_acc_male - avg_acc_female
                print(f"\nAccuracy Difference (Male - StereoFemaletype): {acc_diff}")
                print(f"Average Male Accuracy: {avg_acc_male:.4f}")
                print(f"Average Female Accuracy: {avg_acc_female:.4f}")
                
                
                
                
               
              
                
                
                
                
                
                
                
                
        except FileNotFoundError as e:
            print(f"Warning: Could not evaluate {method} - {str(e)}")
    
   
            
    
    # Print comparison of results
    if results:
        print("\nComparison of Methods:")
        print("-" * 50)
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # Optionally save results
        if args.save_results:
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 
                f"{args.dataset}_{map_model_name(args.model_name)}_{args.embedding_strategy}_comparison.csv")
            df_results.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dama',
                      help='Dataset to evaluate (dama, etc.)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                      help='Model name used for results')
    parser.add_argument('--apply_projection', type=str, default='False',
                      help='Whether projection was applied during evaluation')
    parser.add_argument('--projection_method', type=str, default='LEACE',
                      help='Comma-separated projection methods to evaluate')
    parser.add_argument('--embedding_strategy', type=str, default='mean',
                      choices=['mean', 'last', 'last_non_pad'],
                      help='Embedding strategy used')
    parser.add_argument('--layers', type=str, default='lm_head',
                      help='Layer where projection was applied')
    parser.add_argument('--eval_type', type=str, default='gender_score',
                      help='Type of evaluation to perform')
    parser.add_argument('--split', type=str, default='test',
                      help='Data split to evaluate')
    parser.add_argument('--save_results', type=str, default='False',
                      help='Whether to save comparison results to CSV')

    args = parser.parse_args()

    # Convert string arguments to appropriate types
    args.apply_projection = str_to_bool(args.apply_projection)
    args.save_results = str_to_bool(args.save_results)
    
    main(args)