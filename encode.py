import torch
from pathlib import Path
import numpy as np
import argparse
from model import load_bert_model_tokenizer, ME5Model
from data import get_dataset_handler 
import os
import gc
import h5py
from utils import str_to_bool
from config import SAMPLE_PARAMS, get_sampled_model_name, map_model_name
from transformers import AutoModel, AutoTokenizer
from torch import Tensor


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_E5_embeddings_with_memory_management(model, tokenizer, texts, batch_size, output_file, embedding_type='mean', device='cpu'):
    """
    Generate and save E5 embeddings incrementally to manage memory usage
    Args:
        model: E5 model
        tokenizer: E5 tokenizer
        texts: List of texts to encode
        batch_size: Batch size for processing
        output_file: Path to save embeddings
        embedding_type: Type of embedding to use ('mean' for average pooling)
        device: Device to use for computation
    """
    num_samples = len(texts)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Check if the model is an ME5Model instance
    is_me5_model = isinstance(model, ME5Model)
    
    # Get embedding dimension from a small sample
    with torch.no_grad():
        if is_me5_model:
            # For ME5Model class
            inputs = model.prepare_inputs([texts[0]])
            sample_output = model.get_embeddings(inputs['input_ids'], inputs['attention_mask'])
            embedding_dim = sample_output.shape[-1]
        else:
            # For standard AutoModel
            sample_text = f"query: {texts[0]}"
            sample_input = tokenizer(sample_text, return_tensors="pt")
            sample_input = {k: v.to(device) for k, v in sample_input.items()}
            sample_output = model(**sample_input)
            embedding_dim = sample_output.last_hidden_state.shape[-1]
        
        del sample_output
        torch.cuda.empty_cache() if device == 'cuda' else gc.collect()
    
    # Create HDF5 file to save embeddings incrementally
    with h5py.File(output_file, 'w') as f:
        dset = f.create_dataset('embeddings', shape=(num_samples, embedding_dim), dtype='float32')
        
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f'Processing batch {i//batch_size + 1}/{num_batches}')
            
            # Process single batch
            with torch.no_grad():
                if is_me5_model:
                    # For ME5Model class
                    inputs = model.prepare_inputs(batch_texts)
                    batch_embeddings = model.get_embeddings(inputs['input_ids'], inputs['attention_mask'])
                else:
                    # For standard AutoModel
                    formatted_texts = [f"query: {text}" for text in batch_texts]
                    inputs = tokenizer(formatted_texts, 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=512, 
                                    return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    
                    if embedding_type == 'mean':
                        # Use average_pool function for mean pooling
                        batch_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    
                    # Normalize embeddings
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # Save batch directly to file
                dset[i:i + len(batch_texts)] = batch_embeddings.cpu().numpy()
                
                # Clear memory
                del batch_embeddings
                torch.cuda.empty_cache() if device == 'cuda' else gc.collect()

def get_BERT_embeddings_with_memory_management(model, tokenizer, texts, batch_size, output_file, embedding_type='cls', device='cpu'):
    """
    Generate and save embeddings incrementally to manage memory usage
    """
    num_samples = len(texts)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Get embedding dimension from a small sample
    with torch.no_grad():
        sample_input = tokenizer(texts[0], return_tensors="pt")
        sample_input = {k: v.to(device) for k, v in sample_input.items()}
        sample_output = model(**sample_input, output_hidden_states=True)
        embedding_dim = sample_output.hidden_states[-1].shape[-1]
        del sample_output, sample_input
        torch.cuda.empty_cache() if device == 'cuda' else gc.collect()
    
    # Create HDF5 file to save embeddings incrementally
    with h5py.File(output_file, 'w') as f:
        # Create dataset with final size
        dset = f.create_dataset('embeddings', shape=(num_samples, embedding_dim), dtype='float32')
        
        for i in range(0, num_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f'Processing batch {i//batch_size + 1}/{num_batches}')
            
            # Process single batch
            with torch.no_grad():
                inputs = tokenizer(batch_texts, 
                                padding=True, 
                                truncation=True, 
                                max_length=512, 
                                return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, output_hidden_states=True)
                
                if embedding_type == 'cls':
                    batch_embeddings = outputs.hidden_states[-1][:, 0, :]
                elif embedding_type == 'pooler':
                    batch_embeddings = model.bert(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    ).pooler_output
                elif embedding_type == 'mean':
                    # Use simple mean for BERT, not the average_pool function
                    batch_embeddings = outputs.hidden_states[-1].mean(dim=1)
                
                # Save batch directly to file
                dset[i:i + len(batch_texts)] = batch_embeddings.cpu().numpy()
                
                # Clear memory
                del outputs, inputs, batch_embeddings
                torch.cuda.empty_cache() if device == 'cuda' else gc.collect()

def main(args):
    # Load data based on dataset name
    data_handler = get_dataset_handler(args.dataset)
    if args.dataset == 'multilingual':
        data = data_handler.prepare_data(load_test=True,
                                     to_one_hot=False,
                                     embeddings=False)
    else:
        data = data_handler.prepare_data(load_test=True,
                                     embeddings=False)
    
    # Sample data if requested
    if args.sample_data:
        # Sample training data
        data['X_train'], data['z_train'], data['y_train'] = data_handler.get_sample_data(
            data['X_train'], data['z_train'], data['y_train'],
            n=SAMPLE_PARAMS[args.dataset]['train_size'],
            p_y=args.p_y,
            p_y_z=args.p_y_z,
            seed=SAMPLE_PARAMS['sample_seed']
        )

        # Sample validation data
        data['X_val'], data['z_val'], data['y_val'] = data_handler.get_sample_data(
            data['X_val'], data['z_val'], data['y_val'],
            n=SAMPLE_PARAMS[args.dataset]['val_size'],
            p_y=args.p_y,
            p_y_z=args.p_y_z,
            seed=SAMPLE_PARAMS['sample_seed']
        )
        
        # Sample test data
        data['X_test'], data['z_test'], data['y_test'] = data_handler.get_sample_data(
            data['X_test'], data['z_test'], data['y_test'],
            n=SAMPLE_PARAMS[args.dataset]['test_size'],
            p_y=args.p_y,
            p_y_z=0.5,
            seed=SAMPLE_PARAMS['sample_seed']
        )
    
    # Load model and tokenizer
    model_path = args.model_path if args.base_model == 'multilingual-e5-base' else Path("models") / args.dataset / args.model_path
    
    if args.base_model == 'multilingual-e5-base' or 'e5' in args.base_model.lower():
        print(f"Loading E5 model: {model_path}")
        
        # Check if it's a fine-tuned ME5 model with our custom classifier
        classifier_path = os.path.join(model_path, "classifier_head.pt")
        if os.path.exists(classifier_path):
            # Load as ME5Model
            model = ME5Model.from_pretrained(model_path, device=args.device)
            tokenizer = model.tokenizer
        else:
            print('HERE')
            # Load as standard AutoModel
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.to(args.device)
        
        model.eval()
    else:
        num_labels = data_handler.num_labels if args.base_model == 'bert-base-uncased' else np.unique(data['y_train'])
        model, tokenizer, device = load_bert_model_tokenizer(
                model_name=model_path,
                num_labels=num_labels,
                freeze_base=True,
                freeze_all=True,
                device=args.device
            )
    
    # Generate embeddings for each split
    splits = ['train', 'val', 'test']
    for split in splits:
        texts = data[f'X_{split}']
        
        # Create output directory
        output_dir = Path("data") / "embeddings" / args.dataset
        
        # Handle different paths depending on the model
        if args.base_model == 'multilingual-e5-base' or 'e5' in args.base_model.lower():
            output_dir = output_dir /map_model_name(args.base_model)
        else:
            output_dir = output_dir / args.model_path
        
        # Modify output directory if using sampled data
        if args.sample_data:
            sample_suffix = f"_py{args.p_y}_pyz{args.p_y_z}"
            output_dir = output_dir / f"sampled{sample_suffix}"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        print('Saving embeddings to:', output_dir)
        
        # Define HDF5 output file
        if args.base_model == 'multilingual-e5-base' or 'e5' in args.base_model.lower():
            output_path = output_dir / f"{split}_{args.embedding_type}_embeddings.h5"
            # Generate and save embeddings with memory management for E5
            get_E5_embeddings_with_memory_management(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                batch_size=args.batch_size,
                output_file=str(output_path),
                device=args.device
            )
        else:
            output_path = output_dir / f"{split}_{args.embedding_type}_embeddings.h5"
            # Generate and save embeddings with memory management
            get_BERT_embeddings_with_memory_management(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                batch_size=args.batch_size,
                output_file=str(output_path),
                embedding_type=args.embedding_type,
                device=args.device
            )
        
        print(f"Saved {split} embeddings to {output_path}")
        
        # Clear memory between splits
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                      help="Dataset to use (e.g. bios)")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the finetuned model or HF model name")
    parser.add_argument("--embedding_type", type=str, default="cls",
                      choices=['cls', 'pooler', 'mean'],
                      help="Type of embedding to extract")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for processing")
    parser.add_argument("--device", type=str, 
                      default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for computation")
    # Add sampling arguments
    parser.add_argument("--sample_data", type=str, default='False',
                      help="Sample data to balance classes")
    parser.add_argument("--p_y", type=float, default=0.5,
                      help="P(Y=1) for sampled data")
    parser.add_argument("--p_y_z", type=float, default=0.5,
                      help="P(Y=1|Z=1) for sampled data")
    parser.add_argument("--base_model", type=str, default='bert-base-uncased',
                      help="Base model to use for embeddings")
    
 
    args = parser.parse_args()
    # Convert sample_data string to boolean
    args.sample_data = str_to_bool(args.sample_data)
    main(args)
