import torch
import torch.nn.functional as F
import pandas as pd
from config import TOKEN, map_model_name
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import os, sys
import pickle
from utils import str_to_bool, get_index_of_last_no_padding_token, get_model_id, get_layers_to_process
import gc
from torch.utils.data import Dataset, DataLoader
from data import prepare_tokenized_data
from model import ProjectionLayer, load_projection, ModelWithProj,load_model_and_tokenizer

def get_visible_token(tokenizer, token_id):
    """
    Get a visible representation of a token.
    
    Args:
        tokenizer: The tokenizer
        token_id: The token ID to decode
    
    Returns:
        A visible representation of the token
    """
    # First try standard decode
    token = tokenizer.decode([token_id])
    
    # If empty string, try to get token from vocabulary
    if token == '':
        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            # Get the raw token representation
            vocab_token = tokenizer.convert_ids_to_tokens(token_id)
            token = f"<{vocab_token}>"
        else:
            # Fallback if convert_ids_to_tokens not available
            token = f"<id:{token_id}>"
            
    return token

def get_pred(model, tokenizer, inputs, max_length=None):
    """
    Get predicted next token for a batch of prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        inputs: Either a list of text prompts or a pre-tokenized batch
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Tuple of (predicted_tokens, token_ids, probabilities)
    """
    
    # get the token_id of pad_token
    pad_token_id = tokenizer.pad_token_id

    # Get model's raw predictions with memory optimization
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=False,
    )

    # Get just the logits
    logits = outputs.logits
    
    # Get indices of last non-padding tokens
    last_non_pad_token_indeces = torch.stack([get_index_of_last_no_padding_token(input_ids, pad_token_id) 
                                            for input_ids in inputs['input_ids']]).squeeze(-1)

    # Extract logits at the last non-padding positions
    relevant_logits = torch.stack([logits[i, last_non_pad_token_indeces[i], :] for i in range(len(inputs['input_ids']))])

    # Convert to probabilities - relevant logits is (batch_size, vocab_size)
    probs = relevant_logits.softmax(dim=-1)
    
    # Clean up GPU memory immediately
    del logits, outputs, relevant_logits
    
    # get the max probability
    max_probs, max_indices = probs.max(dim=-1)
    
    # get the token of the max prob - explicitly handle empty string case
    max_tokens = []
    for index in max_indices:
        token_id = index.item()
        token = get_visible_token(tokenizer, token_id)
        max_tokens.append(token)
    
    # Also return the raw token IDs and their probabilities for debugging
    token_ids = max_indices.cpu().tolist()
    token_probs = max_probs.cpu().tolist()
    
    # Clean up
    del probs, max_probs, max_indices
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()
    
    return max_tokens

def get_pred_batched(model, tokenizer, prompts, batch_size=32, max_length=None):
    """
    Calculate predicted next token for a list of prompts in batches using efficient data loading.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of text prompts
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        List of predicted tokens for each prompt
    """
    # Prepare data loader for efficient batching
    dataloader = prepare_tokenized_data(prompts, tokenizer, model.device, batch_size, max_length=max_length)
    print('Tokenized data')
    
    predictions = []
    
    # Process batches
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get predictions for the batch
            batch_predictions = get_pred(model, tokenizer, inputs)
            predictions.extend(batch_predictions)
            
    return predictions

def get_target_prob(model,tokenizer, inputs, target, max_length=None):
    """
    Get probability of target token(s) across a batch of prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        inputs: Either a list of text prompts or a pre-tokenized batch
        target: Target token or list of target tokens to get probability for
        max_length: Maximum sequence length for tokenization
    
    Returns:
        If target is a string: numpy array of probabilities for that target
        If target is a list: list of numpy arrays, one for each target
    """
    # Check if target is a single string or list
    single_target = isinstance(target, str)
    if single_target:
        targets = [target]
    else:
        targets = target
    
    # get the token_id of pad_token
    pad_token_id = tokenizer.pad_token_id

    # Get model's raw predictions with memory optimization
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=False,
    )

    # Get just the logits
    logits = outputs.logits
    
    # Get indices of last non-padding tokens
    last_non_pad_token_indeces = torch.stack([get_index_of_last_no_padding_token(input_ids, pad_token_id) 
                                            for input_ids in inputs['input_ids']]).squeeze(-1)

    # Extract logits at the last non-padding positions
    relevant_logits = torch.stack([logits[i, last_non_pad_token_indeces[i], :] for i in range(len(inputs['input_ids']))])

    # Convert to probabilities - relevant logits is (batch_size, vocab_size)
    probs = relevant_logits.softmax(dim=-1)
    
    # Clean up GPU memory immediately
    del logits, outputs, relevant_logits
    
    # Get target probabilities for each target
    target_probs_list = []
    for t in targets:
        target_id = tokenizer.encode(t, add_special_tokens=False)[0]
        target_probs = probs[:, target_id].cpu().numpy()
        target_probs_list.append(target_probs)
    
    # Final cleanup
    del probs
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
  
    gc.collect()
    
    # Return single array for single target, or list for multiple targets
    if single_target:
        return target_probs_list[0]
    else:
        return target_probs_list


def get_gender_score_batched(model, tokenizer, prompts, batch_size=32, max_length=512):
    """
    Calculate gender score for a list of prompts in batches using efficient data loading.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of text prompts
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (gender_scores, he_probs, she_probs)
    """
    # Prepare data loader for efficient batching
    # prepare_tokenized_data(X, tokenizer, args.device, args.batch_size, max_length=max_length)
    dataloader = prepare_tokenized_data(prompts, tokenizer, model.device, batch_size, max_length=max_length)
    print('Tokenized data')
    
    gender_scores = []
    he_probs = []
    she_probs = []
    
    # Process batches
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc="Processing batches")):

            # Get probabilities for 'he' and 'she' in one call
            he, she = get_target_prob(model, tokenizer, inputs, ["he", "she"])

            # Calculate gender score
            batch_scores = he - she
            gender_scores.extend(batch_scores)
            he_probs.extend(he)
            she_probs.extend(she)
            
    return gender_scores, he_probs, she_probs

def calc_gender_score(model, tokenizer, dataset_path, batch_size, 
                    max_length=None):
    """Process dataset and add model predictions"""
    # Read dataset
    df = pd.read_csv(dataset_path)

    # get the prompts
    prompts = df['prompt'].tolist()

    # If max_length is not provided, calculate it from the prompts
    if max_length is None:
        max_length = max(len(tokenizer.encode(prompt)) for prompt in prompts)
        print(f"Using calculated max_length: {max_length}")
    else:
        print(f"Using provided max_length: {max_length}")

    # turn model to eval mode
    model.eval()
    
    # Calculate gender scores
    print("Calculating gender scores...")
    gender_scores, he_probs, she_probs = get_gender_score_batched(
            model, tokenizer, prompts, batch_size, max_length=max_length)
   

    df['gender_score'] = gender_scores
    df['he_prob'] = he_probs
    df['she_prob'] = she_probs

    return df


def calc_coreference(model, tokenizer, dataset_path, batch_size, 
                    max_length=None):

    """ Read dataset and add model predictions"""
    
    # Read dataset
    df = pd.read_csv(dataset_path)
    
    print(model.lm_head)
    sys.exit()
    
    # get the prompts
    prompts = df['prompt'].tolist()
    
    if max_length is None:
        max_length = max(len(tokenizer.encode(prompt)) for prompt in prompts)
        print(f"Using calculated max_length: {max_length}")
    else:
        print(f"Using provided max_length: {max_length}")
    # turn model to eval mode
    model.eval()
    predicted_tokens = get_pred_batched(
            model, tokenizer, prompts, batch_size, max_length=max_length)
    
    # Add predicted tokens to DataFrame
    df['predicted_tokens'] = predicted_tokens
        
    # Add a column for the profession
    # get the unique professions
    professions_for_tok = df['profession_for_tokenizer'].unique()
    
    # remove nan values
    professions_for_tok_str = [prof for prof in professions_for_tok if isinstance(prof, str)]
    
    # get their tokens
    profession_tokenized =[tokenizer.decode(tokenizer.encode(prof)[1]) for prof in professions_for_tok_str]

    # map the tokens to the professions
    profession_map = {prof: token for token, prof in zip(profession_tokenized, professions_for_tok)}
    df['correct_tok'] = df['profession_for_tokenizer'].map(profession_map)
    
    return df
    

def main(args):    
    # Set device based on args
    if args.device == "auto":
        device_map = "auto"
    else:
        # Use specific device (cuda, cpu, mps)
        device_map = args.device
    
    # Load model and tokenizer
    print(f"Loading model on device: {device_map}")
    model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            model_type=args.model_type,
            task_type="causal-lm" if args.model_type == "llama" else None,
            device_map=args.device,
            torch_dtype=args.torch_dtype
    )


    # if apply_projection is True, then define model as model
    if args.apply_projection:
        model = ModelWithProj(model, model_type=args.model_type)

        projection_params = {
            "model_name": args.model_name,
            "model_type": args.model_type,
            "dataset": args.dataset,
            "projection_method": args.projection_method,
            "layers": get_layers_to_process(args.layers, model, args.model_type),
            "embedding_strategy": args.embedding_strategy
        }
    else:
        projection_params = None


    # Apply projection if requested
    if args.apply_projection and projection_params is not None:

        apply_strategy = args.embedding_strategy if args.apply_strategy == 'same' else args.apply_strategy

        # Handle multiple layers
        for layer_id in projection_params["layers"]:
            # Determine folder structure based on args.layers
            if args.layers == "all":
                layers_folder = "all"
            elif args.layers == "lm_head":
                layers_folder = "lm_head"
            elif args.layers.startswith("last_") and args.layers[5:].isdigit():
                layers_folder = args.layers
            else:
                layers_folder = "custom"
                
            # Load the projection
            projection = load_projection(
                model_name=projection_params["model_name"],
                dataset=projection_params["dataset"],
                device=model.device,
                projection_method=projection_params["projection_method"],
                layer_id=layer_id,
                embedding_strategy=projection_params["embedding_strategy"],
                projections_dir=f"projections/{projection_params['dataset']}",
                layer_folder=layers_folder,
                independent_layers=args.independent_layers
            )
            
             # Determine which layer to register the hook on
            if layer_id != 'lm_head' and args.model_type == "llama":
                # Register projection hook for the next layer
                layer_to_register = layer_id + 1
            else:
                layer_to_register = layer_id
            
            model.register_projection_hook(layer_id=layer_id, projection=projection, apply_strategy=apply_strategy)
    else:
        print("No projection applied")

    
    
    
    # set the dataset path
    if args.dataset == 'dama' or args.dataset == 'dama_mixed':
        if args.dataset == 'dama':
            dataset_path = 'data/dama/dama_professions.csv'
        elif args.dataset == 'dama_mixed':
            dataset_path = 'data/dama/dama_professions_mixed.csv'
        
        # Process dataset
        df = calc_gender_score(
            model, 
            tokenizer, 
            dataset_path, 
            args.batch_size,
            max_length=args.max_length
        )
    elif args.dataset == 'winobias':
        print("Calculating coreference resolution...")
        
        dataset_path = 'data/winobias/winobias.csv'
        
        df = calc_coreference(
            model, 
            tokenizer, 
            dataset_path, 
            args.batch_size,
            max_length=args.max_length
        )
        print("Coreference resolution done")

    # remove hooks
    if args.apply_projection and projection_params:
       model.remove_hooks()
    # map model name
    model_name_short = map_model_name(args.model_name)


    # Modify output path if projection was used
    suffix = ""
    if args.apply_projection:
        if args.layers == "lm_head":
            suffix = f"_{args.projection_method}_lm_head"
        elif args.layers == "all":
            suffix = f"_{args.projection_method}_all"
        elif args.layers.startswith("last_"):
            suffix = f"_{args.projection_method}_{args.layers}"
        else:
            layers_str = "_".join([str(layer) for layer in projection_params["layers"]])
            suffix = f"_{args.projection_method}_{layers_str}"

         # Add embedding and apply strategy to suffix
        independent_str = "independent" if args.independent_layers else "dependent"
        suffix += f"_{args.embedding_strategy}_{independent_str}"
    
    print(f"Model name: {model_name_short}")
    print(f"Projection method: {args.projection_method}")
    print(f"Layers: {args.layers}")
    print(f"Embedding strategy: {args.embedding_strategy}")
    print(f"Apply strategy: {args.apply_strategy}")
    print(f"Independent layers: {args.independent_layers}")
    print(f"Suffix: {suffix}")    
    print('df.head()')
    print(df.head())
    
    # create the output folder if it does not exist
    output_folder = Path(args.output_folder if not None else 'data/result_data')
    if not output_folder.exists():
        print(f"Creating output folder: {output_folder}")
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # define the output path
    output_path = f'{args.output_folder}/{args.dataset}_professions_{model_name_short}{suffix}.csv'
    print(f"Saving results to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Add progress bar to pandas
    tqdm.pandas()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dama", "dama_mixed", "bios", "toy", 'winobias'],
                    help="Dataset to use for gender score calculation")
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                      help='Model name to use')
    parser.add_argument('--model_type', type=str, default='llama',
                      help='Model type (llama or bert)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=None,
                      help='Maximum sequence length for tokenization (calculated from data if not provided)')
    parser.add_argument('--apply_projection', type=str, default='False',
                      help='Whether to apply projection during evaluation')
    parser.add_argument('--projection_method', type=str, default='LEACE',
                      help='Projection method used (LEACE, LEACE-no-whitening, etc.)')
    parser.add_argument('--layers', type=str, default='lm_head',
                      help="Layers to apply projection: 'last', 'all', 'lm_head', 'last_x' (where x is a number), or specific layers like '0,1,2'")
    parser.add_argument('--embedding_strategy', type=str, default='mean',
                      help='Embedding strategy (mean, last, etc.)')
    parser.add_argument("--apply_strategy", type=str, default="all", choices=["all", "last_non_pad","same" ],
                        help="Which layers to apply the projection to")
    parser.add_argument('--device', type=str, default='mps',
                      help='Device to use (auto, cuda, cpu, mps)')
    parser.add_argument("--torch_dtype", type=str, default="float16", 
                        choices=["float16", "float32", "bfloat16"],
                        help="Precision to use for computations")
    parser.add_argument('--independent_layers', type=str, default='False',
                      help='Whether to use independent layers for projection')
    parser.add_argument('--output_folder', default="None", type=str,
                      help='Path where the output CSV file should be saved')
    args = parser.parse_args()
        
    # Convert apply_projection to boolean
    args.apply_projection = str_to_bool(args.apply_projection) 
    args.independent_layers = str_to_bool(args.independent_layers)   
    
    # if output_folder is None, set it to the current directory
    if args.output_folder == "None":
        args.output_folder = None

    main(args)