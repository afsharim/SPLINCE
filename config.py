TOKEN="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"

# Sampling parameters
SAMPLE_PARAMS = {
    'sample_seed': 0,
    'bios': {
    'train_size': 75000,
    'val_size': 10000,
    'test_size': 25000,
    },
    'multilingual': {
    'train_size': 3334,
    'val_size': 446,
    'test_size': 800,
    },
}

LAST_LAYER_RETRAIN_PARAMS = {
      'lr': 0.001,
      'epochs': 5
 }




def map_model_name(model_name):
    if model_name == "meta-llama/Llama-2-7b-hf":
        return "Llama_2_7B"
    elif model_name == "meta-llama/Llama-2-13b-hf":
        return "Llama_2_13B"
    elif model_name == 'meta-llama/Llama-3.1-8B-Instruct':
        return "Llama_3_8B"
    elif model_name == 'meta-llama/Llama-3.1-8B':
        return "Llama_3_8B_orig"
    elif model_name == "bert-base-uncased":
        return "BERT_base"
    elif model_name == "intfloat/multilingual-e5-base":
        return "ME5_base"
    elif model_name == 'mistralai/Mistral-7B-v0.3':  # Add this line
        return "Mistral_7B"
    elif model_name == 'microsoft/phi-2':
        return "Phi_2"
    else:
        return model_name

def get_sampled_model_name(model_name, p_y_z):
    """Add sampling parameters to model name"""
    return f"{model_name}_sampled_pyz{p_y_z}"