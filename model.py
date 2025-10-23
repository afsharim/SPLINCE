import torch
from transformers import (
    LlamaForCausalLM, LlamaTokenizer,
    BertForSequenceClassification, BertTokenizer,
    TrainingArguments, Trainer,
    AutoTokenizer, AutoModel, AutoModelForCausalLM
)
from datasets import Dataset
from config import TOKEN
import os
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AdamW
import torch.nn as nn
from utils import get_torch_dtype
from config import map_model_name
import pickle
from transformers import AutoModelForSequenceClassification


class LastLayer(nn.Module):
    """Linear layer with loaded weights from BERT classifier"""
    def __init__(self, d, num_labels, coef=None, intercept=None):
        super().__init__()
        self.linear = nn.Linear(d, num_labels)
        
        # Initialize with provided weights if given
        if coef is not None:
            self.linear.weight.data = torch.as_tensor(coef, dtype=torch.float16)
        if intercept is not None:
            self.linear.bias.data = torch.as_tensor(intercept, dtype=torch.float16)
    
    def forward(self, x):
        return self.linear(x)

    def predict(self, x, type_pred='class', threshold=0.5):
        # Forward pass
        logits = self.forward(x)

        if type_pred == 'class':
            # check if the model is binary or multiclass
            if logits.shape[1] == 1:
                preds = torch.sigmoid(logits) > threshold
            else:
                preds = torch.argmax(logits, dim=1)
        elif type_pred == 'prob':
            preds = probs
        return preds

    def fit(self, X, y, lr=0.01, optimizer='adam', wd=0.01, epochs=100, batch_size=32):
        """Train the last layer using specified optimizer and parameters"""
        # Convert inputs to tensors if they aren't already
        X = torch.as_tensor(X, dtype=torch.float16)
        y = torch.as_tensor(y, dtype=torch.float16)
        
        
        # Set up optimizer
        if optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Set up loss function - BCEWithLogitsLoss for binary, CrossEntropyLoss for multiclass
        if self.linear.out_features == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        self.train()  # Set model to training mode
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            total_loss = 0
            
            # Process mini-batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]


                # Forward pass
                outputs = self(X_batch)

                if self.linear.out_features == 1:
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(outputs, y_batch.long())

                # Backward pass
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            # Print progress every 10 epochs
            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / n_batches
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        self.eval()  # Set model back to evaluation mode



def extract_bert_classifier(model, dim=None):
    """Extract classifier weights and bias from BERT model"""

    # Extract all weights and bias
    classifier = LastLayer(model.config.hidden_size, model.num_labels, model.classifier.weight.data, model.classifier.bias.data)
   
    return classifier

def load_model_and_tokenizer(
    model_name, 
    model_type="auto", 
    task_type=None, 
    num_labels=None,
    device_map="auto", 
    torch_dtype=None,
    token=TOKEN
):
    """
    General function to load model and tokenizer with flexible options.
    
    Args:
        model_name: str, name or path of the model in HuggingFace format
        model_type: str, specific model type or "auto" for automatic detection
        task_type: str, optional task type (e.g., "causal-lm", "sequence-classification")
        num_labels: int, number of labels for classification tasks
        device_map: str or dict, device mapping strategy
        torch_dtype: str or torch.dtype, precision for model weights
        load_in_8bit: bool, whether to load model in 8-bit precision
        load_in_4bit: bool, whether to load model in 4-bit precision
        token: str, HuggingFace token for gated models
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Convert string torch_dtype to actual torch dtype if needed
    if isinstance(torch_dtype, str):
        torch_dtype = get_torch_dtype(torch_dtype)
    
    # Configure tokenizer args
    tokenizer_kwargs = {}
    model_kwargs = {
        "torch_dtype": torch_dtype}
    
    if device_map != "cpu" and device_map is not None:
        model_kwargs["device_map"] = device_map
    
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    
    
    # Handle different model types
    if model_type == "llama" or (model_type == "auto" and "llama" in model_name.lower()):
        print(f"Loading LLaMA model: {model_name} with the following kwargs: {model_kwargs}")
       
        # check; if llama3, use AutoTokenizer
        if model_name=='meta-llama/Llama-3.1-8B' or model_name=='meta-llama/Llama-3.1-8B-Instruct':
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if task_type == "sequence-classification" or num_labels is not None:
            # Not yet handling LLaMA classification, would require additional setup
            raise NotImplementedError("LLaMA for classification not implemented yet")
        else:
            model = LlamaForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    elif model_type == "bert" or (model_type == "auto" and "bert" in model_name.lower()):
        print(f"Loading BERT model: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        if task_type == "sequence-classification" or num_labels is not None:
            model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,

                **model_kwargs
            )
        else:
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
    
    else:
        # Use Auto classes for other model types
        print(f"Loading model with Auto classes: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if task_type == "causal-lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        elif task_type == "sequence-classification" and num_labels is not None:
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels,
                **model_kwargs
            )
        else:
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
    
    # Move model to specific device if device_map is a string (not "auto")
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0"]:
        model = model.to(device_map)
    
    return model, tokenizer


def load_adapted_model_tokenizer(original_model_name, model_type, path,device_map="auto", torch_dtype=torch.float16):

    # based on the model type, determine the tokenizer
    if model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(original_model_name, token=TOKEN)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    # load the model
    if model_type == 'llama':
        model = LlamaForCausalLM.from_pretrained(
            path,
            device_map=args.device,
            torch_dtype=torch.float16,
            token=TOKEN
        )
    
    return model, tokenizer

def load_bert_model_tokenizer(model_name="bert-base-uncased", num_labels=None, freeze_base=False, freeze_all=False, device="cpu", torch_dtype=torch.float16):
    """Load BERT model and tokenizer for classification. Now uses the general function."""
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        model_type="bert",
        task_type="sequence-classification",
        num_labels=num_labels,
        device_map=device,
        torch_dtype=torch_dtype  # Use default
    )
    
    if freeze_base:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze classification head parameters
        if not freeze_all:
            for param in model.classifier.parameters():
                param.requires_grad = True
            
        print("Model parameters frozen except classification head")
        # Print number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")



    
        
    return model, tokenizer, device  # Return device as well

def compute_metrics(pred):
    """Compute accuracy for evaluation"""
    with torch.no_grad():  # Ensure no gradients are stored
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

 # Prepare datasets
def tokenize_function_BERT(examples):
    return tokenizer.batch_encode_plus(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

def finetune_bert(
    model,
    tokenizer,
    train_texts,
    train_labels,
    optimizer_type="adamw",
    val_texts=None,
    val_labels=None,
    output_dir="checkpoints",
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    device="cpu",  # Add device parameter
    load_best_at_end=True
):
    """Finetune BERT model on classification task"""
    
   

    print('Total n of train_texts:', len(train_texts))
    print('Total n of val_texts:', len(val_texts))

    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels
    })


    # Prepare datasets
    def tokenize_function_BERT(examples):
        return tokenizer.batch_encode_plus(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    
    # Process Train dataset with specified batch_size
    train_dataset = train_dataset.map(
        tokenize_function_BERT,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train data",
        num_proc=4
    )

    # Create & Process validation dataset with same batch_size
    if val_texts is not None and val_labels is not None:

        val_dataset = Dataset.from_dict({
            "text": val_texts,
            "label": val_labels
        })
        val_dataset = val_dataset.map(
            tokenize_function_BERT,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing validation data",
            num_proc=4
        )
    else:
        val_dataset = None

    # Set format without explicit tensor conversion
    train_dataset.set_format(type='torch')
    if val_dataset:
        val_dataset.set_format(type='torch')

    # put the model on the device
    model.to(device)

    # if optimizer_type == "adamw":
    if optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    # Define training arguments with CUDA optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
        save_steps=1,
        eval_steps=1 if val_dataset else None,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=load_best_at_end,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        use_mps_device=True if device == "mps" else False,
        gradient_accumulation_steps=1,
        dataloader_pin_memory=True if device in ['cuda', 'mps'] else False
    )

    class CustomTrainer(Trainer):
        def evaluation_loop(self, *args, **kwargs):
            with torch.no_grad():  # Ensure no gradients during evaluation
                return super().evaluation_loop(*args, **kwargs)
       

    # Initialize trainer with metrics
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add metrics computation
        optimizers=(optimizer, None)  # Pass optimizer to trainer
    )

    # Train model
    train_result = trainer.train()
    
    # If we have validation data, get the best model
    if load_best_at_end and val_dataset is not None:
        # Load the best model
        best_model_path = os.path.join(output_dir, "checkpoint-best")
        if os.path.exists(best_model_path):
            model = BertForSequenceClassification.from_pretrained(best_model_path)
            print(f"Loaded best model from {best_model_path}")
        else:
            print("Warning: Best model checkpoint not found")
    
    return model, tokenizer, train_result.metrics


class ME5Model(nn.Module):
    """
    ME5 model with classification head for fine-tuning.
    This class wraps the base ME5 model and adds a classification layer on top
    of the pooled embeddings.
    """
    
    def __init__(self, model_name, num_labels, device='cuda'):
        super().__init__()
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device)
        self.base_model.to(self.device)
        
        # Classification head
        self.embedding_dim = self.base_model.config.hidden_size
        self.classifier = nn.Linear(self.embedding_dim, num_labels)
        self.classifier.to(self.device)
        
        # Number of labels determines loss function
        self.num_labels = num_labels
        if num_labels == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
            print("Using BCEWithLogitsLoss for binary classification")
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss for multi-class classification")
    
    def average_pool(self, last_hidden_states, attention_mask):
        """
        Apply average pooling to hidden states using the attention mask.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def prepare_inputs(self, texts):
        """
        Add 'query:' prefix to texts and tokenize them.
        """
        formatted_texts = [f"query: {text}" for text in texts]
        encodings = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encodings.items()}
    
    def get_embeddings(self, input_ids, attention_mask):
        """
        Get normalized average-pooled embeddings from the model.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.average_pool(outputs.last_hidden_state, attention_mask)
        # Normalize embeddings
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def forward(self, input_ids=None, attention_mask=None, texts=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Pre-tokenized input IDs
            attention_mask: Attention mask for input IDs
            texts: Raw text inputs (alternative to input_ids/attention_mask)
            labels: Optional labels for calculating loss
            
        Returns:
            dict: Contains logits and optional loss
        """
        # Handle text inputs if provided
        if texts is not None:
            inputs = self.prepare_inputs(texts)
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        
        # Get embeddings
        embeddings = self.get_embeddings(input_ids, attention_mask)
        
        # Get logits from classifier
        logits = self.classifier(embeddings)
        
       
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # For binary classification
                loss = self.loss_fn(logits, labels.float())
            else:
                # For multi-class classification
                loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'embeddings': embeddings
        }
    
    def save_pretrained(self, output_dir):
        """
        Save the model and tokenizer to output_dir.
        """
        self.base_model.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), f"{output_dir}/classifier_head.pt")
    
    @classmethod
    def from_pretrained(cls, model_dir, num_labels=None, device='cuda'):
        """
        Load a pretrained ME5Model.
        """
        # Load base model and tokenizer
        base_model = AutoModel.from_pretrained(model_dir)
        
        # Determine number of labels from saved classifier if not provided
        if num_labels is None:
            classifier_path = f"{model_dir}/classifier_head.pt"
            if os.path.exists(classifier_path):
                state_dict = torch.load(classifier_path, map_location=device)
                num_labels = state_dict['weight'].shape[0]
            else:
                raise ValueError("Unable to determine num_labels. Please specify it explicitly.")
        
        # Create model instance
        model = cls(model_dir, num_labels=num_labels, device=device)
        
        # Load classifier weights if they exist
        classifier_path = f"{model_dir}/classifier_head.pt"
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        
        return model

def load_me5_model_tokenizer(model_name, num_labels=None, device='cuda'):
    """
    Load an ME5 model and tokenizer.
    
    Args:
        model_name: Name or path of the ME5 model
        num_labels: Number of classification labels
        device: Device to load the model onto
    
    Returns:
        model, tokenizer, device
    """
    print(f"Loading ME5 model: {model_name} with num_labels={num_labels} on device={device}")
    
    # Check if this is a fine-tuned model or base model
    classifier_path = f"{model_name}/classifier_head.pt"
    if os.path.exists(classifier_path):
        # For fine-tuned models
        model = ME5Model.from_pretrained(model_name, num_labels=num_labels, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        # For base models
        base_model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if num_labels is not None:
            model = ME5Model(model_name, num_labels=num_labels, device=device)
        else:
            # Return just the base model if num_labels not provided
            model = base_model.to(device)
    
    return model, tokenizer, device

def finetune_me5(model, tokenizer, train_texts, train_labels, val_texts=None, val_labels=None, 
                output_dir=None, batch_size=16, num_epochs=3, learning_rate=2e-5, 
                weight_decay=0.01, device='cuda', optimizer_type='adamw'):
    """
    Finetune an ME5 model for classification tasks.
    
    Args:
        model: ME5 model or base model
        tokenizer: ME5 tokenizer
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        output_dir: Output directory
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to use
        optimizer_type: Type of optimizer to use
    
    Returns:
        model, tokenizer, metrics
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    import numpy as np
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Check if we need to wrap the model in ME5Model
    if not isinstance(model, ME5Model):
        num_labels = len(torch.unique(train_labels))
        is_binary = num_labels <= 2
        if is_binary and len(train_labels.shape) == 1:
            num_labels = 1
        me5_model = ME5Model(model_name=None, num_labels=num_labels, device=device)
        me5_model.base_model = model
        model = me5_model
    
    # Format texts with "query:" prefix and create datasets
    train_inputs = model.prepare_inputs(train_texts)
    train_dataset = TensorDataset(
        train_inputs['input_ids'],
        train_inputs['attention_mask'],
        train_labels.to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_texts is not None and val_labels is not None:
        val_inputs = model.prepare_inputs(val_texts)
        val_dataset = TensorDataset(
            val_inputs['input_ids'],
            val_inputs['attention_mask'],
            val_labels.to(device)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Setup learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_metric = 0
    best_model_state = None
    metrics = {
        "train_loss": [], 
        "val_loss": [], 
        "val_acc": [], 
        "val_f1": [],
        "val_precision": [],
        "val_recall": []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        metrics["train_loss"].append(train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                    val_loss += loss.item()
                    
                    # Get predictions
                    if model.num_labels == 1:  # Binary classification with sigmoid
                        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().flatten()
                    else:  # Multi-class with softmax
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            val_loss /= len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Store metrics
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)
            metrics["val_f1"].append(val_f1)
            metrics["val_precision"].append(val_precision)
            metrics["val_recall"].append(val_recall)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model based on F1 score (or accuracy for simpler tasks)
            val_metric = val_f1  # Can be changed to val_acc if preferred
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_model_state = {
                    'base_model': model.base_model.state_dict().copy(),
                    'classifier': model.classifier.state_dict().copy()
                }
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
    
    # Load best model if available
    if best_model_state is not None:
        model.base_model.load_state_dict(best_model_state['base_model'])
        model.classifier.load_state_dict(best_model_state['classifier'])
    
    # Save model if output_dir provided
    if output_dir:
        model.save_pretrained(output_dir)
    
    return model, tokenizer, metrics


class ProjectionLayer:
    """Class to handle loading and applying projection matrices"""
    def __init__(self, P, b, device, dtype=torch.float16):
        """Initialize with projection matrix P and optional bias b"""
        self.P = P
        self.b = b

        # if P is a numpy array, convert to torch tensor
        if isinstance(self.P, np.ndarray):
            self.P = torch.from_numpy(self.P)
        
        # if b is a numpy array, convert to torch tensor
        if isinstance(self.b, np.ndarray):
            self.b = torch.from_numpy(self.b)

        # Convert to specified dtype
        self.P = self.P.to(dtype)
        self.b = self.b.to(dtype)


        # Move to device
        self.P = self.P.to(device)
        self.b = self.b.to(device)

    
    def apply(self, x):
        """Apply projection to input tensor x"""
        # Convert to tensor if numpy array
        x_proj =  x @ self.P.T + self.b

        return x_proj


class ModelWithProj(nn.Module):
    def __init__(self, model, model_type="llama"):
        """
        Args:
            model: A pre-initialized Hugging Face model (e.g. Llama 2 7B)
            model_type: A string indicating the model type ("llama" or "bert")
        """
        super(ModelWithProj, self).__init__()
        self.model = model
        self.model_type = model_type
        self.current_attention_mask = None  # This will be set during forward
        self._projection_hooks = []
        self.device = model.device

        # if model_type == "llama":
        if model_type == "llama":
            self.base_model = model.base_model
            self.lm_head = model.lm_head
            self.layers = model.base_model.layers
        elif model_type == "bert":
            self.bert = model.bert
            self.layers = model.bert.encoder.layer
       

    def forward(self, *args, **kwargs):
        # Capture the attention mask from kwargs (if provided)
        if "attention_mask" in kwargs:
            self.current_attention_mask = kwargs["attention_mask"]
        # Forward through the wrapped model
        return self.model(*args, **kwargs)

     # Define the hook function as a closure capturing `self`
    def proj_hook(self, module, input_tensors, projection, apply_strategy):
        
        # Get the hidden states from the hook input (first positional argument)
        hidden_states = input_tensors[0]

        # Access the attention mask from the top-level model
        attn_mask = self.current_attention_mask

        # apply_strategy can be "last", "last_non_pad", "all", or "cls"
        if apply_strategy == "last":
            selected_state = hidden_states[:, -1]
        elif apply_strategy == "last_non_pad":
            if attn_mask is None:
                raise ValueError("Attention mask is required for 'last_non_pad' apply strategy")
            # Determine the index of the last non-pad token per example
            last_indices = attn_mask.sum(dim=1).long() - 1
            batch_size = hidden_states.shape[0]
            selected_state = hidden_states[:, last_indices]
        elif apply_strategy == "all":
            selected_state = hidden_states
        elif apply_strategy == "cls":
            selected_state = hidden_states[:, 0]
        else:
            raise ValueError(f"Unsupported apply strategy: {apply_strategy}")
    
        # Apply the projection
        modified_states = projection.apply(selected_state)

        # Put the modified states back based on the apply_strategy
        if apply_strategy == "last":
            input_tensors[0][:, -1] = modified_states
        elif apply_strategy == "last_non_pad":
            input_tensors[0][:, last_indices] = modified_states
        elif apply_strategy == "all":
            input_tensors[0][:, :] = modified_states
        elif apply_strategy == "cls":
            input_tensors[0][:, 0] = modified_states
  
        return input_tensors

    def register_projection_hook(self, layer_id, projection, apply_strategy="last_non_pad"):
        """
        Registers a forward pre-hook on a specified layer to apply a projection.
        
        Args:
            layer_id: Identifier for the target layer ("lm_head" or an integer index).
            projection: An object with an `apply` method to project hidden states.
            apply_strategy: Strategy for selecting which token's embedding to use.
        
        Returns:
            hook_handle: A handle to the registered hook (to remove later if needed).
        """
       

        # Determine the target module based on the model type and layer identifier
        if layer_id == "lm_head":
            target_module = self.model.lm_head
        else:
            target_module = self.layers[layer_id]
      

        # Register the forward pre-hook on the target module
        hook_handle = target_module.register_forward_pre_hook(
            lambda module, input_tensors: self.proj_hook(
                module, input_tensors, projection, apply_strategy
            )
        )
        self._projection_hooks.append(hook_handle)
        return hook_handle

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._projection_hooks:
            handle.remove()
        self._projection_hooks = []




def load_projection(
    model_name, 
    dataset, 
    device,
    projection_method="LEACE", 
    layer_id="lm_head", 
    embedding_strategy="mean",
    projections_dir="projections",
    layer_folder="lm_head",
    independent_layers=True
):
    """
    Load projection matrix and bias for a specific model and dataset.
    
    Args:
        model_name: Name of the model
        dataset: Dataset name
        projection_method: Method used to create projection
        layer_id: Layer identifier
        embedding_strategy: Strategy used for creating
        projections_dir: Base directory where projections are stored
        
    Returns:
        ProjectionLayer object with P and b
    """
    # Convert layer_id to string for path construction
    layer_str = str(layer_id)
    
    # Use map_model_name for consistency with calc_projections.py
    model_name_short = map_model_name(model_name)
    
    # Construct path to projection directory
    independent_folder = "independent" if independent_layers else "dependent"
    projection_dir = os.path.join(
        projections_dir,
        f"{model_name_short}_{projection_method}_{embedding_strategy}",
        f"{layer_folder}",
        independent_folder,
        f"layer_{layer_str}"
    )
    
    if not os.path.exists(projection_dir):
        raise FileNotFoundError(f"Projection directory not found: {projection_dir}")
    
    # Load projection matrix and bias
    P_path = os.path.join(projection_dir, "P.npy")
    b_path = os.path.join(projection_dir, "b.npy")
    P = np.load(P_path)
    b = np.load(b_path)

    # Load metadata for verification
    metadata_path = os.path.join(projection_dir, "metadata.pkl")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded projection for {metadata['model_name_short'] if 'model_name_short' in metadata else metadata['model_name']}, layer {metadata['layer_id']}")
    
    return ProjectionLayer(P, b, device=device)




def normalize_p_dict(p):
    """
    Normalize the proportion of each group to sum to 1
    
    Arguments:
        p: dictionary with the proportion of each group
    
    Returns:
        p: dictionary with the proportion of each group normalized to sum to 1
    """
    # normalize each entry in p to sum to 1
    p_sum = sum(p.values())
    p = {g: p[g] / p_sum for g in p.keys()}
    return p


def get_q(loss_g, eta, eps=10**-5):
    """
    Get the q for a group g, where the q is the exponential of the loss multiplied by the learning rate

    Arguments:
        loss_g: float, loss for group g
        eta: float, learning rate
        eps: float, small number to avoid numerical instability
    
    Returns:
        q: float, q for group g
    """
    q = np.exp(eta * (loss_g + eps))
    return q


def calc_BCE(y_true, y_pred):
    """
    Calculate binary cross entropy loss
    
    Arguments:
        y_true: true labels
        y_pred: predicted probabilities
        
    Returns:
        loss: binary cross entropy loss
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def calc_worst_group_loss(model, loss_fn, X, y, g):
    """
    Calculate worst group loss and loss per group
    
    Arguments:
        model: trained model
        loss_fn: loss function
        X: features
        y: labels
        g: group assignments
        
    Returns:
        worst_loss: worst group loss
        loss_dict: dictionary with loss per group
    """
    groups = np.unique(g)
    loss_dict = {}
    
    for group in groups:
        group_idx = (g == group)
        if np.sum(group_idx) > 0:
            X_group = X[group_idx]
            y_group = y[group_idx]
            
            # Get predictions - assuming model has predict method that returns probabilities
            if hasattr(model, 'predict_proba'):
                y_pred_group = model.predict_proba(X_group)[:, 1]  # Get positive class probabilities
            else:
                # Fallback for custom models
                y_pred_group = model.predict(X_group, type_pred='probabilities').flatten()
            
            loss_dict[int(group)] = loss_fn(y_group, y_pred_group)
    
    worst_loss = max(loss_dict.values()) if loss_dict else 0.0
    return worst_loss, loss_dict


class GDROModel:
    """Simplified Group Distributionally Robust Optimization model wrapper."""
    
    def __init__(self, base_model):
        """
        Initialize GDRO model
        
        Arguments:
            base_model: sklearn classifier (e.g., SGDClassifier)
        """
        self.base_model = base_model
        self.group_weights = {}
        self.classes_initialized = False
        self.all_classes = None
        
    def assign_weights(self, g):
        """
        Assign weights to samples based on group membership
        
        Arguments:
            g: group assignments
            
        Returns:
            weights: sample weights
        """
        weights = np.zeros(len(g))
        for group, weight in self.group_weights.items():
            weights[g == group] = weight
        return weights
    
    def reset_weights(self, q_dict):
        """
        Reset group weights
        
        Arguments:
            q_dict: dictionary with weights for each group
        """
        self.group_weights = q_dict.copy()
    
    def get_Beta(self):
        """
        Combine the coef and intercept in Beta
        """
        coef = self.base_model.coef_
        intercept = self.base_model.intercept_
        return np.concatenate((intercept, coef[0])).reshape(-1, 1)
    
    def gradient_step_DRO(self, X_b, y_b, g_b, eta_param):
        """
        Run a single gradient step for the sklearn model
        """
        # get the weights
        w_b = self.assign_weights(g_b)

        # set the learning rate
        self.base_model.learning_rate = 'constant'
        self.base_model.eta0 = eta_param

        # use the partial fit method of the classifier
        # Handle the case where not all classes are present in the batch
        if not self.classes_initialized:
            # First call - use all unique classes from the batch
            unique_classes = np.unique(y_b)
            self.all_classes = unique_classes
            self.classes_initialized = True
            self.base_model.partial_fit(X_b, y_b, sample_weight=w_b, classes=self.all_classes)
        else:
            # Subsequent calls - always use the same class set to avoid the error
            self.base_model.partial_fit(X_b, y_b, sample_weight=w_b, classes=self.all_classes)

        # turn the coef_ and intercept_ into Beta
        self.Beta = self.get_Beta()
    
    def optimize_GDRO_via_SGD(self, X_train, y_train, g_train, X_val, y_val, g_val, 
                             T, batch_size, eta_param, eta_q, C=0.0, early_stopping=False, patience=1):
        """
        Run the GDRO algorithm for the model
        
        Arguments:
            X_train, y_train, g_train: training data
            X_val, y_val, g_val: validation data
            T: number of epochs
            batch_size: batch size
            eta_param: learning rate for parameters
            eta_q: learning rate for group weights
            C: regularization parameter
            early_stopping: whether to use early stopping
            patience: patience for early stopping
            
        Returns:
            best_Beta: best parameters
            best_worst_group_loss: best worst group loss
            best_t: best epoch
        """
        # get the number of groups in the training data
        groups = np.unique(g_train)

        # get the initial q_t
        q_t = {int(group): 1/len(groups) for group in groups}

        # set the initial weights
        self.reset_weights(q_t)

        # define the n_dict; this is the number of observations in each group
        n_dict = {int(group): np.sum(g_train == group) for group in groups}

        # save the best worst-case loss
        best_worst_group_loss = np.inf

        # loss function
        loss_fn = calc_BCE
        
        # Initialize the classifier with all possible classes from the full training set
        # This prevents issues when batches don't contain all classes
        self.all_classes = np.unique(y_train.squeeze(-1) if y_train.ndim > 1 else y_train)
        print(f"Initializing GDRO with classes: {self.all_classes}")

        # loop over T epochs
        for t in range(T):
            
            # shuffle the indices
            indices = np.arange(X_train.shape[0])
            shuffled_indices = np.random.permutation(indices)

            # loop over the batches, including the last batch
            for i in range(0, len(shuffled_indices), batch_size):

                # get the final index
                last_batch_index = min(i+batch_size, len(shuffled_indices))

                # get the batch indices
                batch_indices = shuffled_indices[i:last_batch_index]

                # get the batch
                X_b, y_b, g_b = X_train[batch_indices, :], y_train[batch_indices], g_train[batch_indices]
                
                # Ensure y_b is properly shaped and check for class coverage
                y_b_flat = y_b.squeeze(-1) if y_b.ndim > 1 else y_b
                batch_classes = np.unique(y_b_flat)
                
                # Skip batches that are too small or have issues
                if len(batch_classes) == 0 or len(X_b) == 0:
                    continue

                # update the parameters
                self.gradient_step_DRO(X_b, y_b_flat, g_b, eta_param)

                # get loss per group for the batch
                _, loss_dict = calc_worst_group_loss(self, loss_fn, X_b, y_b_flat, g_b)
                
                # based on the loss, update the q_t
                q_t = update_DRO_weights(q_t, loss_dict, eta_q, C=C, n_dict=n_dict)

                # reset the group weights of the model
                self.reset_weights(q_t)

            # after completing the first epoch, get the param
            if t == 0:
                best_Beta = self.Beta
                best_t = t
            
            # after completing an epoch, get the loss for the entire dataset
            y_val_flat = y_val.squeeze(-1) if y_val.ndim > 1 else y_val
            worst_group_loss_val, _ = calc_worst_group_loss(self, loss_fn, X_val, y_val_flat, g_val)
                    
            # print the following stats
            print('At epoch {}, the worst group loss on the validation set is {} '.format(t, worst_group_loss_val))

            if early_stopping and worst_group_loss_val >= best_worst_group_loss:
                patience -= 1
                if patience == 0:
                    print('Early stopping at epoch {}'.format(t))
                    break
            
            # if loss is better, save the parameters
            if worst_group_loss_val < best_worst_group_loss:
                best_worst_group_loss = worst_group_loss_val
                best_Beta = self.Beta
                best_t = t
                
        print('Best worst group loss ({}) found at epoch {}'.format(best_worst_group_loss, best_t))
        return best_Beta, best_worst_group_loss, best_t
    
    def predict(self, X, type_pred='probabilities'):
        """
        Make predictions using the base model
        
        Arguments:
            X: features
            type_pred: type of prediction ('probabilities' or 'class')
            
        Returns:
            predictions
        """
        if type_pred == 'probabilities':
            return self.base_model.predict_proba(X)[:, 1]
        elif type_pred == 'class':
            return self.base_model.predict(X)
        else:
            raise ValueError("type_pred must be 'probabilities' or 'class'")


def update_DRO_weights(q, loss_dict, eta_q, C=0.0, n_dict=None, p_min=0.0, p_max=1.0): 
    """
    Update the weights for each group using the DRO update rule

    Arguments:
        q: dictionary with the weights for each group
        loss_dict: dictionary with the loss for each group
        eta_q: float, learning rate
        C: float, regularization parameter
        n_dict: dictionary with the number of samples for each group
        p_min: float, minimum weight for each group
        p_max: float, maximum weight for each group
    
    Returns:
        q_updated: dictionary with the updated weights for each group
    """
    # get the groups in the loss dict
    groups_loss = list(loss_dict.keys())

    # go through each group and update the weights
    for g in groups_loss:
        
        # Get the loss for group g
        loss_g = loss_dict[g]

        if n_dict is not None:
            n_g = n_dict[g]
            regularizer = C/np.sqrt(n_g)
        else:
            regularizer = 0.0
        
        # Get the q for group g
        q_g = get_q(loss_g, eta_q) + regularizer

        # Update the weights for group g
        q[g] *= q_g
    
    # Normalize the weights
    q_normalized = normalize_p_dict(q)

    return q_normalized
