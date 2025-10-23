import argparse
from pathlib import Path
import json
from config import map_model_name, SAMPLE_PARAMS, get_sampled_model_name
from data import DamaData, BiosData
from model import (
    load_bert_model_tokenizer,
    finetune_bert,
    load_me5_model_tokenizer,
    finetune_me5,
    ME5Model
)
from utils import str_to_bool, set_seed
import sys
import torch
from data import get_dataset_handler
import numpy as np

def finetune(args):
    # get dataset handler
    data_handler = get_dataset_handler(args.dataset)
    if args.dataset == 'multilingual':
        data = data_handler.prepare_data(load_test=False,
                                     to_one_hot=False,
                                     embeddings=False)
    else:
        data = data_handler.prepare_data(load_test=False,
                                     embeddings=False)
    
    X_train, z_train, y_train = data['X_train'], data['z_train'], data['y_train']
    X_val, z_val, y_val = data['X_val'], data['z_val'], data['y_val']

    # Sample data if requested
    if args.sample_data:
        # Sample training data
        X_train, z_train, y_train = data_handler.get_sample_data(
            X_train, z_train, y_train,
            n=SAMPLE_PARAMS[args.dataset]['train_size'],
            p_y=args.p_y,
            p_y_z=args.p_y_z,
            seed=SAMPLE_PARAMS['sample_seed']
        )

        # Sample validation data
        X_val, z_val, y_val = data_handler.get_sample_data(
            X_val, z_val, y_val,
            n=SAMPLE_PARAMS[args.dataset]['val_size'],
            p_y=args.p_y,
            p_y_z=args.p_y_z,
            seed=SAMPLE_PARAMS['sample_seed']
        )
    
    print('shape of z, y:', z_train.shape, y_train.shape)


    # Convert labels to tensors
    num_labels = np.unique(y_train).shape[0]
    if args.model_type == 'me5' and num_labels ==2:
        num_labels = 1
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.int64)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.int64)
   

    # Load model and tokenizer based on model type
    if args.model_type == 'me5':
        model, tokenizer, device = load_me5_model_tokenizer(
            model_name=args.model_name,
            num_labels=num_labels,
            device=args.device
        )
    else:  # Default is BERT
        model, tokenizer, device = load_bert_model_tokenizer(
            model_name=args.model_name,
            num_labels=num_labels,
            device=args.device,
            freeze_base=False,
            freeze_all=False,
            torch_dtype=torch.float32 if args.model_name == 'bert-base-uncased' else torch.float16
        )
    
    # Modify output directory name if using sampled data
    model_name_short = map_model_name(args.model_name)
    if args.sample_data:
        model_name_short = get_sampled_model_name(model_name_short, args.p_y_z)
    
    output_dir = Path("models") / args.dataset / f"{model_name_short}_bs{args.batch_size}_lr{args.learning_rate}_e{args.num_epochs}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Finetune model based on model type
    set_seed(args.seed)
    if args.model_type == 'me5':
        model, tokenizer, metrics = finetune_me5(
            model=model,
            tokenizer=tokenizer,
            train_texts=X_train,
            train_labels=y_train_tensor,
            val_texts=X_val,
            val_labels=y_val_tensor,
            output_dir=str(output_dir),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            optimizer_type=args.optimizer_type
        )
    else:  # Default is BERT
        model, tokenizer, metrics = finetune_bert(
            model=model,
            tokenizer=tokenizer,
            optimizer_type=args.optimizer_type,
            train_texts=X_train,
            train_labels=y_train_tensor,
            val_texts=X_val,
            val_labels=y_val_tensor,
            output_dir=str(output_dir),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device
        )

    # Save model and tokenizer
    if args.model_type == 'me5' and hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Save training parameters
    training_params = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "num_train_examples": len(X_train),
        "num_val_examples": len(X_val) if X_val is not None else None,
        "num_classes": num_labels,
        "weight_decay": args.weight_decay,
        "final_metrics": metrics,
    }
    
    params_path = output_dir / f"training_params.json"
    with open(params_path, "w") as f:
        json.dump(training_params, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bios",
                      help="Dataset to finetune on")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                      help="Base model to finetune")
    parser.add_argument("--model_type", type=str, default="bert",
                      choices=["bert", "me5"],
                      help="Type of model to finetune (bert or me5)")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument('--optimizer_type', type=str, default='adamw',
                      help='Optimizer type')
    parser.add_argument("--output_dir", type=str, default="models",
                      help="Output directory for finetuned model")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for regularization")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use for training (cpu, cuda, or mps)")
    
    # Add sampling arguments
    parser.add_argument("--sample_data", type=str, default='False',
                      help="Sample data to balance classes")
    parser.add_argument("--p_y", type=float, default=0.5,
                      help="P(Y=1) for sampled data")
    parser.add_argument("--p_y_z", type=float, default=0.5,
                      help="P(Y=1|Z=1) for sampled data")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for training")
    
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    
    # change sample_data to boolean
    args.sample_data = str_to_bool(args.sample_data)
    
    # finetune
    finetune(args)
   
if __name__ == "__main__":
    main()
