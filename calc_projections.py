import argparse
import os
import pickle
import numpy as np
import torch
from tqdm.auto import tqdm
import pandas as pd
import sys
from proj import proj
from utils import set_seed, get_torch_dtype, get_index_of_last_no_padding_token, get_layers_to_process, str_to_bool, est_Cov
from data import get_dataset_handler, prepare_tokenized_data
from model import load_model_and_tokenizer, load_bert_model_tokenizer
import gc
import importlib
from config import map_model_name
from model import ProjectionLayer,  load_projection, ModelWithProj


def extract_layer_activations(model, dataloader, layer_id, model_type, device, model_dtype, embedding_strategy="mean", pad_token_id=0):
    """
    Extract activations from specific layer.
    
    Args:
        model: The model to extract activations from
        dataloader: DataLoader with the tokenized inputs
        layer_id: Which layer to extract (integer index or "lm_head")
        model_type: Type of model ("llama" or "bert")
        device: Computing device
        embedding_strategy: How to handle sequence dimension ("mean", "last")
        pad_token_id: ID of padding token (for "last_non_pad" strategy)
    
    Returns:
        numpy.ndarray: Activations of shape (batch_size, hidden_dim)
    """
    # Initialize list to store activations
    activations = []
    
    # Set model to eval mode
    model.eval()

    # do not store gradients 
    with torch.no_grad():

        # Process each batch
        for batch in tqdm(dataloader, desc=f"Extracting layer {layer_id}"):
            # Get inputs
            inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device)  # Convert attention mask to same dtype as model
                }
                            
            # Forward pass and extract activations based on model type and layer_id
            if model_type in ["llama", 'mistral', 'phi']:
                # Get model's parameter dtype for proper conversion
                model_dtype = next(model.parameters()).dtype

                # Complete forward pass and get embeddings before lm_head
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )

                if layer_id == "lm_head":
                    # Get the last hidden state (input to lm_head)
                    hidden_states = outputs.hidden_states[-1].to(model_dtype)
                    
                    # Apply final normalization if available
                    if hasattr(model, "norm"):
                        hidden_states = model.norm(hidden_states)
                    elif hasattr(model, "model") and hasattr(model.model, "norm"):
                        hidden_states = model.model.norm(hidden_states)
                else:
                    # LLaMA model hidden states include embedding layer at index 0
                    # So we need to get the output of layer layer_id+1
                    if isinstance(layer_id, int) and layer_id < len(outputs.hidden_states) - 1:
                        # Get hidden states of target layer
                        hidden_states = outputs.hidden_states[layer_id + 1].to(model_dtype)  # +1 to skip embedding layer
                    else:
                        # Default to last layer if out of range
                        hidden_states = outputs.hidden_states[-1].to(model_dtype) #up to the target layer

                # Process sequence dimension based on strategy
                if len(hidden_states.shape) > 2:  # If we have sequence dimensioneach layer up to the target
                    if embedding_strategy == "mean":
                        # Average over sequence dimension
                        hidden_states = hidden_states.mean(dim=1)
                    elif embedding_strategy == "last":
                        # Take last token representation   
                        hidden_states = hidden_states[:, -1, :]
                    elif embedding_strategy == 'last_non_pad':
                        # Get indices of last non-padding tokens   
                        last_non_pad_token_indeces = torch.stack([get_index_of_last_no_padding_token(input_ids, pad_token_id) for input_ids in inputs['input_ids']]).squeeze(-1) # Remove extra dimension
                        
                        # Extract hidden states of last non-padding tokens
                        hidden_states = torch.stack([hidden_states[i, last_non_pad_token_indeces[i], :] for i in range(len(inputs['input_ids']))])

            
            elif model_type == "bert":

                # define the hook
                layer_input = None
                def get_activation(name):
                    def hook(module, input_):
                        nonlocal layer_input
                        layer_input = input_[0].detach().to(model_dtype)   # Store output and detach
                    return hook
                
                # Register the hook for the specified layer in the BERT model
                model.layers[layer_id-1].register_forward_pre_hook(get_activation(layer_id))

                # Process through embedding
                _ = model(input_ids=inputs['input_ids'],
                                     attention_mask=inputs['attention_mask'],
                                     output_hidden_states=False)

                # get the hidden state
                hidden_states = layer_input
                        
                # gather the hidden states of the CLS token
                if embedding_strategy == "cls":
                    hidden_states = hidden_states[:, 0, :]

           

                
            # Collect activations and clear memory
            activations.append(hidden_states)
    
            
        # Concatenate all batcheselif device == "mps" and hasattr(torch.mps, 'empty_cache'):
        all_activations = torch.cat(activations, dim=0).cpu().numpy()  # Convert to numpy array

         # Clear GPU memory based on device type
        if device == "cuda" or (isinstance(device, str) and "cuda" in device):
            torch.cuda.empty_cache() 
       
        # Explicit clean-up
        del inputs, hidden_states, activations
        gc.collect()


    return all_activations



def calculate_projections(args):
    """Main function to calculate and save projections."""
    
    # Set random seed for reproducibility,
    set_seed(args.seed)

    # Get dataset handler for metadatadding_strategy
    data_handler = get_dataset_handler(args.dataset)            
    attribute = data_handler.attribute_name
    target = data_handler.target_name    
    
    if args.sample_data:
        print('Sampling data with p(y|z):', args.p_y_z)
        data_dict = data_handler.prepare_data(
                load_test=True, 
                embeddings=False, 
                model_name=args.model_name,
                single_y=True,  # Add this to get binary labels,
                sample=args.sample_data,
                p_y_z=args.p_y_z,
                p_y=0.5
            )
    else:
        
        # one-hot encode y if args.dataset is winobias
        if args.dataset == "winobias":
            data_dict = data_handler.prepare_data( one_hot_y=True)
        else:
            data_dict = data_handler.prepare_data()
        
        

    model_kwargs = {
                    "model_name": args.model_name,
                    "model_type": args.model_type,
                    "task_type": "causal-lm" if args.model_type in ["llama", 'mistral'] else None,
                    "device_map": args.device,
                    "torch_dtype": args.torch_dtype
                }
    # if the model-type is bert
    if args.model_type == "bert":
        model, tokenizer, device = load_bert_model_tokenizer(
            model_name=args.model_name,
            num_labels=data_handler.num_labels,
            freeze_base=True,
            freeze_all=True,
            device=args.device,
            torch_dtype=args.torch_dtype
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,  
            model_type=args.model_type,
            task_type="causal-lm" if args.model_type in ["llama", "mistral"] else None,
            device_map=args.device,
            torch_dtype=args.torch_dtype
        )

    # Add projection layer to model
    model = ModelWithProj(model, model_type=args.model_type)

    # Extract training data for projection calculations
    X = data_dict['X_train']
    z = data_dict['z_train']
    y = data_dict['y_train']

    # Prepare output directory 
    model_name_short = map_model_name(args.model_name)
    
    # Create folder name based on layers argument
    if args.layers == "all":
        layers_folder = "all"
    elif args.layers == "lm_head":
        layers_folder = "lm_head"
    elif args.layers.startswith("last_") and args.layers[5:].isdigit():
        layers_folder = args.layers  # Use "last_x" directly as folder name
    else:
        layers_folder = args.layers
    
    independent_folder = "independent" if args.independent_layers else "dependent"
    output_base_dir = os.path.join(
        args.output_dir, 
        args.dataset,
        f"{model_name_short}_{args.projection_method}_{args.embedding_strategy}",
        independent_folder,
        layers_folder
    )
    os.makedirs(output_base_dir, exist_ok=True)
    print('Output directory:', output_base_dir)
    

    # Determine which layers to processz = data_dict['z_train']
    layers_to_process = get_layers_to_process(args.layers, model, args.model_type)
    
    # save the handles
    hook_handles = []
    
    # Process text data if necessary
    if isinstance(X[0], str):
        # Use smaller sequence length if requestedint(l.strip()) for l in args.layers.split(",")]
        max_length = args.max_length if hasattr(args, 'max_length') else 512
        dataloader = prepare_tokenized_data(X, tokenizer, args.device, args.batch_size, max_length=max_length)
        
        # Create projection
        projector = proj()
        
        # Process each layer
        for layer_id in layers_to_process:
            layer_str = str(layer_id)       
            print(f"Processing layer {layer_str}...")       

            # Get output directory for this layer
            # This creates the directory structure: {output_base_dir}/layer_{layer_str}/# Create projection
            layer_output_dir = os.path.join(output_base_dir, f"layer_{layer_str}")
            os.makedirs(layer_output_dir, exist_ok=True)

            # Dictionary to store projections for each layer
            # Extract activations for this layer{}
            activations = extract_layer_activations(
                model, dataloader, layer_id, args.model_type, args.device,
                embedding_strategy=args.embedding_strategy,
                pad_token_id=tokenizer.pad_token_id,
                model_dtype=get_torch_dtype(args.torch_dtype)
            )
            layer_output_dir = os.path.join(output_base_dir, f"layer_{layer_str}")
           

            # Fit projection
            projector.fit(activations, z, y, method=args.projection_method)
                        
            # if args.calculate_cov
            if args.calculate_cov:

                # Calculate cov(X, z) and cov(X, y) matrices
                cov_Xz = est_Cov(activations, z)
                cov_Xy = est_Cov(activations, y)
                
                # Calculate the activations, after projection
                activations_proj = projector.apply_projection(activations, y)
                
                # Calculate cov(X_proj, z) and cov(X_proj, y) matrices
                cov_Xz_proj = est_Cov(activations_proj, z)
                cov_Xy_proj = est_Cov(activations_proj, y)
                

                # Save covariance matrices
                # calculate the norm per y
                if cov_Xy.shape[1] > 1:
                    cov_Xy_norm = np.linalg.norm(cov_Xy, axis=1)
                    cov_Xy_proj_norm = np.linalg.norm(cov_Xy_proj, axis=1)
                else:
                    cov_Xy_norm = np.linalg.norm(cov_Xy)
                    cov_Xy_proj_norm = np.linalg.norm(cov_Xy_proj)
                
                # Save covariance matrices
                # calculate the norm per z
                if cov_Xy.shape[1] > 1:
                    cov_Xz_norm = np.linalg.norm(cov_Xz, axis=1)
                    cov_Xz_proj_norm = np.linalg.norm(cov_Xz_proj, axis=1)
                else:
                    cov_Xz_norm = np.linalg.norm(cov_Xz)
                    cov_Xz_proj_norm = np.linalg.norm(cov_Xz_proj)
                    
                
                cov_stats ={
                    "cov_Xz": cov_Xz,
                    "cov_Xy": cov_Xy,
                    "cov_Xz_proj": cov_Xz_proj,
                    "cov_Xy_proj": cov_Xy_proj,
                    "||cov_Xz_proj||": cov_Xz_proj_norm,
                    "||cov_Xy_proj||": cov_Xy_proj_norm,
                    "||cov_Xz||": cov_Xz_norm,
                    "||cov_Xy||": cov_Xy_norm,

                }
                inprod = np.matmul(cov_Xz.T, cov_Xy)
                print('Inproduct:', inprod)
                print('Avg. inproduct:', np.mean(inprod))
            
                print('Covariance norm after:', cov_Xz_proj_norm, cov_Xy_proj_norm)
                print('Covariance norm before:', cov_Xz_norm, cov_Xy_norm)
                
                # save in same location as the projection
                with open(os.path.join(layer_output_dir, "cov_stats.pkl"), "wb") as f:
                    pickle.dump(cov_stats, f)
            
          
            # Save projection matrix and biasdel_type, args.device,
            np.save(os.path.join(layer_output_dir, "P.npy"), projector.P)
            np.save(os.path.join(layer_output_dir, "b.npy"), projector.b)

            # Save metadata
            metadata = {   # Fit projection
                "model_name": args.model_name,
                "model_name_short": model_name_short,  # Save short name in metadata
                "layer_id": layer_id,
                "projection_method": args.projection_method, # Save projection matrix and bias
                "embedding_strategy": args.embedding_strategy,
                "dataset": args.dataset,
                "attribute": attribute,
                "target": target,
            }
            
            with open(os.path.join(layer_output_dir, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
                

            # Cache this projection for use in subsequent layersataset,
            print('Projection at layer', layer_id, 'calculated and saved with dimensions:', projector.P.shape, projector.b.shape, ' and device:', model.device)

            # if more than one layer, and not determined independently
            if len(layers_to_process) > 1 and not args.independent_layers:
                
                # Create projection layer object
                projection = ProjectionLayer(projector.P, projector.b, args.device, get_torch_dtype(args.torch_dtype))

                # create model that can apply projection
                apply_strategy = args.embedding_strategy if args.apply_strategy == 'same' else args.apply_strategy
                
                if layer_id != 'lm_head' and args.model_type == "llama":
                    # Register projection hook for the layer
                    layer_to_register = layer_id +1
                else:
                    layer_to_register = layer_id
                print('The layer calculated for {} is applied to layer {}'.format(layer_id, layer_to_register))
                model.register_projection_hook(layer_id=layer_to_register, projection=projection, apply_strategy=apply_strategy)
                
            

    print("All projections calculated and saved.")

    # remove hooks
    if len(layers_to_process) > 1:
        model.remove_hooks()

   
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Calculate projections for language models")

    # Model parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "bert", "mistral", 'phi'],         
                        help="Model architecture type")
       
    # Projection parametersojections calculated and saved.")
    parser.add_argument("--projection_method", type=str, default="LEACE", 
                        choices=["LEACE", "LEACE-no-whitening", "opt-sep-proj", "SAL"],
                        help="Projection method to use")
    
    # Layer selection
    parser.add_argument("--layers", type=str, default="last",
                        help="Layers to adapt: 'last', 'all', 'lm_head', 'last_x' (where x is a number), or specific layers like '0,1,2'")
    
    # Whether or not to determine layers independently
    parser.add_argument("--independent_layers", type=str, default="False",
                        help="Whether to determine layers independently")

    # Dataset parametersparser.add_argument("--model_type", type=str, default="llama", choices=["llama", "bert"],
    parser.add_argument("--dataset", type=str, required=True, choices=["dama", "dama_mixed", "bios", "toy", "winobias"],
                        help="Dataset to use for projection calculation")
    
    # Sample data parameters
    parser.add_argument("--sample_data", type=str, default="False",
                        help="Whether to sample data with specific properties")
    parser.add_argument("--p_y_z", type=float, default=0.5,
                        help="Conditional probability p(y|z) for sampled data")
    
    # Hardware parameters default="LEACE", 
    parser.add_argument("--device", type=str, default="cpu",                      
                        help="Device to use (cuda, mps, cpu)") 
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--torch_dtype", type=str, default="float16",choices=["float16", "float32", "bfloat16"],
                        help="Precision to use for computations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/llms",
                        help="Directory to save projections")
    # Add embedding/apply strategy parameter
    parser.add_argument("--embedding_strategy", type=str, default="mean", 
                        choices=["mean", "last",  "cls", "first", "last_non_pad"],
                        help="Strategy for handling sequence dimension in embeddings")
    parser.add_argument("--apply_strategy", type=str, default="all", choices=["all", "last_non_pad", 'same', 'cls' ],
                        help="Which layers to apply the projection to")

    # Add memory management parametersparser.add_argument("--seed", type=int, default=42,
    parser.add_argument("--max_length",  type=int, default=512,
                        help="Maximum sequence length for tokenization")

    # whether to calculate the cov(X, z) and cov(X, y) matrices
    parser.add_argument("--calculate_cov", type=str, default="False",
                        help="Whether to calculate cov(X, z) and cov(X, y) matrices")

    args = parser.parse_args()

    args.calculate_cov = str_to_bool(args.calculate_cov)
    args.sample_data = str_to_bool(args.sample_data)
    args.independent_layers = str_to_bool(args.independent_layers)
    calculate_projections(args)

