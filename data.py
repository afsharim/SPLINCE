from pathlib import Path
import json
from abc import ABC, abstractmethod
import pickle
import math
import numpy as np
import h5py
from utils import set_seed
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from config import SAMPLE_PARAMS
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import sys

def get_dataset_handler(dataset_name):
    """Get the appropriate dataset handler based on dataset name."""
    if dataset_name == "dama" or dataset_name == "dama_mixed":
        return DamaData(dataset_name)
    elif dataset_name == "bios":
        return BiosData()
    elif dataset_name == "toy":
        return ToyData()
    elif dataset_name == "multilingual":
        return MultiLingualData()
    elif dataset_name == "winobias":
        return WinobiasData(dataset_name)
    elif dataset_name == "Waterbirds":
        return WaterbirdsData()
    elif dataset_name == "CelebA":
        return CelebAData()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")



def prepare_tokenized_data(texts, tokenizer, device, batch_size, max_length=512):
    """Tokenize text data and create a DataLoader."""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Create dataset from texts
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    # Convert to torch format
    tokenized_dataset.set_format(type="torch", device=device)
    
    # Create dataloader
    dataloader = DataLoader(tokenized_dataset, 
                        batch_size=batch_size)
    
    return dataloader


class Data(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_dir = Path('data') / dataset_name.lower()
        
    @abstractmethod
    def prepare_data(self):
        """Prepare dataset specific data"""
        pass
    
    @property
    def attribute_name(self):
        """Get the name of the protected attribute"""
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    def target_name(self):
        """Get the name of the target variable to preserve"""
        raise NotImplementedError("Subclasses must implement this property")
    
    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save prepared dataset to csv"""
        df.to_csv(output_path, index=False)
        print(f"Prepared dataset saved to {output_path}")

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)


    def demean_X(self, X_train, X_val, X_test):
        """
        Demean the data
        """

        # estimate the mean based on the training data
        X_train_mean = np.mean(X_train, axis=0)

        # demean the data based on the mean estimated from the training data 
        X_train_demeaned = X_train - X_train_mean
        X_val_demeaned = X_val - X_train_mean
        X_test_demeaned = X_test - X_train_mean

     
        return X_train_demeaned, X_val_demeaned, X_test_demeaned

    def unit_var_X(self, X_train, X_val, X_test):
        """
        Set the variance of the data to 1
        """

        # estimate the standard deviation based on the training data
        X_train_std = np.std(X_train, axis=0)

        # set the standard deviation to 1
        X_train_unit_var = X_train / X_train_std
        X_val_unit_var = X_val / X_train_std
        X_test_unit_var = X_test / X_train_std

        return X_train_unit_var, X_val_unit_var, X_test_unit_var

    
    def standardize_X(self, X_train, X_val, X_test):
        """
        Apply standardization to the data (demean and set variance to 1)
        """

        # first, demean the data
        X_train_demeaned, X_val_demeaned, X_test_demeaned = self.demean_X(X_train, X_val, X_test)

        # second, set the variance of the data to 1
        X_train_standard, X_val_standard, X_test_standard = self.unit_var_X(X_train_demeaned, X_val_demeaned, X_test_demeaned)

        return X_train_standard, X_val_standard, X_test_standard

    def to_one_hot(self, y, num_classes=None):
        """Convert class vector (integers) to binary class matrix (one-hot encoding).
        
        Args:
            y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
            num_classes: total number of classes. If None, computed from y.
        
        Returns:
            A binary matrix representation of the input.
        """
        y = np.array(y, dtype=np.int32).flatten()
        if num_classes is None:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        
        # check: if y starts at 1, then we need to subtract 1
        if np.min(y) == 1:
            y = y - 1
        # define one-hot encoded
        categorical[np.arange(n), y] = 1
        
        return categorical

    def get_group(self, y, z):
        """
        Determine group based on y and z values.
        Args:
            y: label (1 or 0)
            z: protected attribute (1 or 0)
        Returns:
            Group identifier (1-4)
        """
        # if y == 1 and z == 1
        if y == 1 and z == 1:
            g = 1
        # if y == 1 and z == 0
        elif y == 1 and z == 0:
            g = 2
        # if y == 0 and z == 1
        elif y == 0 and z == 1:
            g = 3
        # if y == 0 and z == 0
        else:
            g = 4
        return g
    
    def sample_by_probabilities(self, y, z, p_y, p_y_z, n, seed):
        """
        Sample indices to achieve desired probabilities with specific sample size
        Args:
            y: binary labels array
            z: binary protected attribute array 
            p_y: desired P(Y=1)
            p_y_z: desired P(Y=1|Z=1)
            n: desired total sample size
        Returns:
            indices to sample
        """
        # vectorize such that we can get if for each sample
        get_group_v = np.vectorize(self.get_group)

        # Get current group assignments
        groups = get_group_v(y, z)

        # Calculate desired counts for each group
        n_y1 = int(n * p_y)
        n_y0 = n - n_y1
        n_y1_z1 = int(n_y1 * p_y_z)
        n_y1_z0 = n_y1 - n_y1_z1
        n_y0_z0 = int(n_y0 * p_y_z)
        n_y0_z1 = n_y0 - n_y0_z0

        
        # Sample from each group
        g1_idx = np.where(groups == 1)[0]  # Y=1,Z=1
        g2_idx = np.where(groups == 2)[0]  # Y=1,Z=0
        g3_idx = np.where(groups == 3)[0]  # Y=0,Z=1
        g4_idx = np.where(groups == 4)[0]  # Y=0,Z=0
        
        # Handle cases where we don't have enough samples in a group
        n_y1_z1 = min(n_y1_z1, len(g1_idx))
        n_y1_z0 = min(n_y1_z0, len(g2_idx))
        n_y0_z1 = min(n_y0_z1, len(g3_idx))
        n_y0_z0 = min(n_y0_z0, len(g4_idx))
        
        # Sample required numbers from each group
        set_seed(seed)
        idx_g1 = np.random.choice(g1_idx, size=n_y1_z1, replace=False) if n_y1_z1 > 0 else []
        idx_g2 = np.random.choice(g2_idx, size=n_y1_z0, replace=False) if n_y1_z0 > 0 else []
        idx_g3 = np.random.choice(g3_idx, size=n_y0_z1, replace=False) if n_y0_z1 > 0 else []
        idx_g4 = np.random.choice(g4_idx, size=n_y0_z0, replace=False) if n_y0_z0 > 0 else []
        
        # Combine all indices
        indices = np.concatenate([idx_g1, idx_g2, idx_g3, idx_g4])
        np.random.shuffle(indices)
        
        return indices

    def get_y_z_sample(self, y_train, y_val, y_test, z_train, z_val, z_test, p_y_z, p_y=0.5):

        # get the index
        if self.dataset_name == 'bios':
            y_train, y_val, y_test = y_train[:, self.index_y], y_val[:,  self.index_y], y_test[:, self.index_y]

        # get indices for train
        indices_train = self.sample_by_probabilities(
            y_train,
            z_train,
            p_y_z=p_y_z,
            p_y=p_y,
            n=SAMPLE_PARAMS[self.dataset_name]['train_size'],
            seed=SAMPLE_PARAMS['sample_seed']
        )
        
        # get indices for val
        indices_val = self.sample_by_probabilities(
            y_val,
            z_val,
            p_y_z=p_y_z,
            p_y=p_y,
            n=SAMPLE_PARAMS[self.dataset_name]['val_size'],
            seed=SAMPLE_PARAMS['sample_seed']
        )

        # get indices for test
        indices_test = self.sample_by_probabilities(
            y_test,
            z_test,
            p_y_z=0.5, 
            p_y=p_y,
            n=SAMPLE_PARAMS[self.dataset_name]['test_size'],
            seed=SAMPLE_PARAMS['sample_seed']
        )

        # get the sampled data
        y_train, z_train = y_train[indices_train], z_train[indices_train]
        y_val, z_val = y_val[indices_val], z_val[indices_val]
        y_test, z_test = y_test[indices_test], z_test[indices_test]

        # print the number of samples in each group
        g = np.vectorize(self.get_group)
        groups_train = g(y_train, z_train)
        print('Train:', np.unique(groups_train, return_counts=True))
        print('Val:', np.unique(g(y_val, z_val), return_counts=True))
        print('Test:', np.unique(g(y_test, z_test), return_counts=True))
        
        
        # set the indices
        self.indices_train = indices_train
        self.indices_val = indices_val
        self.indices_test = indices_test

        return y_train, z_train, y_val, z_val, y_test, z_test
    
class CelebAData(Data):
    def __init__(self):  
        super().__init__('celebA')
        self.data_dir = 'data/CelebA'
        self.image_dir= 'data/CelebA/images'
        
    def turn_BW_transform(self, target_resolution=(50, 50)):
        # set the target resolution
        transform_func = transforms.Compose([transforms.Grayscale(),
                                                transforms.Resize(target_resolution), 
                                                transforms.ToTensor(),
                                                torch.flatten])
        return transform_func
    
    @property
    def attribute_name(self):
        return 'Smiling'
    @property
    def target_name(self):
        return 'Eyeglasses'
    
    def prepare_data(self):
        pass
    
    def load_df(self, path):
        df = pd.read_csv(path, sep='\s+', skiprows=1)
        df = df.reset_index(drop=False)
        df.columns = ['img_filename'] + list(df.columns[1:])
        
        return df
    
    def transform_images(self, image_filenames, transform_func):
        
        X = []
        for img_filename in image_filenames:
            img_path = self.image_dir +'/'+ img_filename
            img = Image.open(img_path)
            img_processed = transform_func(img)
            X.append(img_processed)
        
        # Stack processed images
        X_final = torch.stack(X)
        
        # turn the X to numpy
        X_final = X_final.numpy()
        
        return X_final
    
    def create_sample(self, n, p_y=0.5, p_y_z=0.5, target_resolution=(50, 50)):
        """
        Create a sample from CelebA dataset with specific probabilities for target and protected attributes.
        
        Args:
            p_y: desired P(Y=1) for 
            p_y_z: desired P(Y=1|Z=1) 
            target_resolution: desired resolution for the images
        
        Returns:
            X: processed images (black and white, flattened)
            y: target labels (Eyeglasses)
            z: protected attribute labels (Smiling)
        """
        # Read the CelebA metadata
        metadata_path = self.data_dir +'/'+ 'list_attr_celeba.txt'
        df_metadata = self.load_df(metadata_path)
        
        # Select necessary columns
        df = df_metadata[['img_filename', self.target_name, self.attribute_name]]
        
        # Convert labels from -1/1 to 0/1
        df[self.target_name] = df[self.target_name].apply(lambda x: 1 if x == 1 else 0)
        df[self.attribute_name] = df[self.attribute_name].apply(lambda x: 1 if x == 1 else 0)
        
        # Get y and z arrays
        y = df[self.target_name].values
        z = df[self.attribute_name].values
        
        # Sample indices according to desired probabilities
        indices = self.sample_by_probabilities(
            y=y,
            z=z,
            p_y=p_y,
            p_y_z=p_y_z,
            n=n,
            seed=SAMPLE_PARAMS['sample_seed']
        )
        
        # Select sampled rows from metadata
        df_sampled = df.iloc[indices]
        
        # Initialize transform
        transform_func = self.turn_BW_transform(target_resolution)
        
        # Transform images
        X_final = self.transform_images(df_sampled['img_filename'], transform_func)
        
        # Get final y and z arrays
        y_final = df_sampled[self.target_name].values.reshape(-1, 1)
        z_final = df_sampled[self.attribute_name].values.reshape(-1, 1)
        
        return X_final, y_final, z_final


    def prepare_data(self, p_y_z, seed, target_res=(50, 50), train_split=0.8, val_split=0.1, **kwargs):

        # load the data
        n = 11000
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        n_test = n - n_train - n_val
        
        # create samples
        X_train, y_train, z_train = self.create_sample(n_train, p_y_z=p_y_z, target_resolution=target_res)
        X_val, y_val, z_val = self.create_sample(n_val, p_y_z=p_y_z, target_resolution=target_res)
        X_test, y_test, z_test = self.create_sample(n_test, p_y_z=0.5, target_resolution=target_res)
        
        # combine X, y, z into a dictionary 
        return {
            'X_train': X_train, 'z_train': z_train, 'y_train': y_train,
            'X_val': X_val, 'z_val': z_val, 'y_val': y_val,
            'X_test': X_test, 'z_test': z_test, 'y_test': y_test
        }
        
        
        
        
   

    

class WinobiasData(Data):
    def _init__(self):
        super().__init__('winobias')
        
            
        
    @property
    def attribute_name(self):
        return "anti_stereotype"
    
    @property
    def target_name(self):
        return "profession_for_tokenizer"
    
    
    def prepare_data(self, one_hot_y=False):
        
        # load the csv
        df = pd.read_csv(self.data_dir / 'winobias.csv')
        
        # Split data into train and test
        df_train = df[df['split'] == 'train']
        df_test = df[df['split'] == 'test']


        # Extract inputs and concepts for training
        X_train = df_train['prompt'].tolist()
        z_train = df_train[self.attribute_name].values.reshape(-1, 1)  # stereotype_score
        y_train = df_train[self.target_name].values # fact_score

        # Extract inputs and concepts for testing
        X_test = df_test['prompt'].tolist()
        z_test = df_test[self.attribute_name].values.reshape(-1, 1)
        y_test = df_test[self.target_name].values   
        
        # if one_hot_y is True, turn y to one-hot encoding
        if one_hot_y:
            
            # create mapping between professions and integers
            unique_professions = np.unique(y_train)
            mapping = {p: i for i, p in enumerate(unique_professions)}
        
            # first, turn y to integers
            y_train = np.array([mapping[p] for p in y_train.flatten()])
            y_test = np.array([mapping[p] for p in y_test.flatten()])
            
            # second, one-hot encode
            y_train = self.to_one_hot(y_train)
            y_test = self.to_one_hot(y_test)
                    
        # Set validation data to None (Winobias doesn't use validation set)
        X_val, z_val, y_val = None, None, None
        
        # Return data dictionary with same structure as load_dataset
        data = {
            'X_train': X_train, 'z_train': z_train, 'y_train': y_train,
            'X_val': X_val, 'z_val': z_val, 'y_val': y_val,
            'X_test': X_test, 'z_test': z_test, 'y_test': y_test
        }
        
        return data
        
        

class DamaData(Data):
    def __init__(self, dataset_name='dama'):
        super().__init__(dataset_name)
    
    @property
    def attribute_name(self):
        """DAMA dataset uses stereotype score as the protected attribute"""
        return "stereotype_score"
    
    @property
    def target_name(self):
        """DAMA dataset uses fact score as the target to preserve"""
        return "fact_score"
    
    
    def extract_profession(self, prompt):
        """Extract profession from prompt like 'The nurse laughed because'"""
        words = prompt.split()
        profession_words = []
        verbs = ['laughed', 'wanted', 'went', 'wished', 'said', 'stayed', 'cried', 'ran', 'ate', 'desired', 'whispered']
        
        for word in words[1:]:  # Skip 'The'
            if word in verbs:
                break
            profession_words.append(word.lower())
        
        return "_".join(profession_words)
    
    def create_dataset(self ):

        # load professions.json, test_dama.json, train_dama.json
        professions = self.load_json(self.data_dir / 'professions.json')
        test_prompts = self.load_json(self.data_dir / 'test_dama.json')
        train_dama = self.load_json(self.data_dir / 'train_dama.json')

        # turn professions into a dictionary
        professions = {entry[0]: (entry[1], entry[2]) for entry in professions}

        # get the prompts in train and test
        train_prompts = [d['prompt'] for d in train_dama]
        prompts = train_prompts + test_prompts
        original_split = ['train'] * len(train_prompts) + ['test'] * len(test_prompts)

        # go over each prompt and assign a profession
        profession_list = []
        for prompt in prompts:
            profession = self.extract_profession(prompt)
            profession_list.append(profession)
        
        # define the factual and stereotype scores
        stereotype_scores = [professions[prof][1] for prof in profession_list]
        fact_scores = [professions[prof][0] for prof in profession_list]

        # create a dataframe
        df = pd.DataFrame({
            'prompt': prompts,
            'profession': profession_list,
            'stereotype_score': stereotype_scores,
            'fact_score': fact_scores,
            'split': original_split
        })

        return df

        
   
    
    def prepare_data(self):
        """
        Prepare the data for the DAMA dataset.
        """
        # Determine which CSV file to use based on the dataset name
        if self.dataset_name == "dama_mixed":
            csv_file = "data/dama/dama_professions_mixed.csv"
        else:  # default to regular dama
            csv_file = "data/dama/dama_professions.csv"
        
        # Read data
        df = pd.read_csv(csv_file)
        
        # Split data into train and test
        df_train = df[df['split'] == 'train']
        df_test = df[df['split'] == 'test']

        # check: there are unique professions in both train and test
        professions_train = set(df_train['profession'])
        professions_test = set(df_test['profession'])
        assert len(professions_train.intersection(professions_test)) == 0
        

        # Extract inputs and concepts for training
        X_train = df_train['prompt'].tolist()
        z_train = df_train[self.attribute_name].values.reshape(-1, 1)  # stereotype_score
        y_train = df_train[self.target_name].values.reshape(-1, 1)    # fact_score

        # Extract inputs and concepts for testing
        X_test = df_test['prompt'].tolist()
        z_test = df_test[self.attribute_name].values.reshape(-1, 1)
        y_test = df_test[self.target_name].values.reshape(-1, 1)

        # Set validation data to None (DAMA doesn't use validation set)
        X_val, z_val, y_val = None, None, None
        
        # Return data dictionary with same structure as load_dataset
        data = {
            'X_train': X_train, 'z_train': z_train, 'y_train': y_train,
            'X_val': X_val, 'z_val': z_val, 'y_val': y_val,
            'X_test': X_test, 'z_test': z_test, 'y_test': y_test
        }
        
        return data

class MultiLingualData(Data):
    def __init__(self, dataset_name='multilingual'):
        super().__init__(dataset_name)

        self.languages_for_sample = ['en', 'de', 'fr']
        self.languages_to_int = {'en': 1, 'de': 2, 'fr': 3, 'es': 4, 'ja': 5, 'ar': 6, 'uk': 7, 'hi': 8}
        self.num_labels = 2
    
    @property
    def attribute_name(self):
        """Multilingual dataset uses language as the protected attribute"""
        return "lang"
    
    @property
    def target_name(self):
        """Multilingual dataset uses toxicity as the target to preserve"""
        return "toxic"

    def load_embeddings(self, model_name, split, embedding_type, p_y_z=0.5, p_y=0.5):

        # get the embeddings directory
        embedding_name = 'sampled_py{}_pyz{}'.format(p_y, p_y_z)

        embeddings_dir = 'data/embeddings/{}/{}/{}'.format(self.dataset_name, model_name ,embedding_name)
        embeddings_dir_split = embeddings_dir + '/' + '{}_{}_embeddings.h5'.format(split, embedding_type)
        print('Loading embeddings from:', embeddings_dir_split)

        # train/val/test embeddings from h5 file
        with h5py.File(embeddings_dir_split, 'r') as f:
            X = f['embeddings'][:]

        # load the requisite labels
        dataset = load_from_disk(self.data_dir / split)

        # Extract language attributes (protected attribute)
        z = dataset['lang']

        # turn to an integer
        z = np.array([self.languages_to_int[lang] for lang in z])
        ints = [self.languages_to_int[lang] for lang in self.languages_for_sample]

        # select only the languages we want to sample from
        mask = np.isin(z, ints)

        # Extract toxicity labels (target variable)
        y = np.array(dataset['toxic'])

        # filter y, z
        y = y[mask]
        z = z[mask]

        # Convert arrays to numpy and reshape if needed
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

        
        return X, z, y


    def load_multilingual(self, split):

        # Load the datasets from disk
        dataset = load_from_disk(self.data_dir / split)

        # Extract text inputs
        X = dataset['text']

        # Extract language attributes (protected attribute)
        z = dataset['lang']
        
        # Extract toxicity labels (target variable)
        y = dataset['toxic']

        # Convert arrays to numpy and reshape if needed
        y = np.array(y).reshape(-1, 1)
        
        return X, z, y


    def prepare_data(self, load_test=False, embeddings=False, embedding_type='pooler', model_name=None, p_y_z=0.5, p_y=0.5, sample=True, to_one_hot=True, **kwargs):
        """
        Load the multilingual dataset from disk and prepare it for use.
        Returns a dictionary with train, validation, and test data.
        """

        # Load embeddings if specified
        if embeddings:
            # Load train and validation embeddings
            X_train, z_train, y_train = self.load_embeddings(model_name, 'train', p_y_z=p_y_z, p_y=p_y, embedding_type=embedding_type)
            X_val, z_val, y_val = self.load_embeddings(model_name, 'val', p_y_z=p_y_z, p_y=p_y, embedding_type=embedding_type)

            
            # Load test data if needed
            if load_test:
                X_test, z_test, y_test = self.load_embeddings(model_name, 'test', p_y_z=p_y_z, p_y=p_y, embedding_type=embedding_type)

            print('X_test shape, z_test shape, y_test shape:', X_test.shape, z_test.shape, y_test.shape)

            # get y, z for the sample
            y_train, z_train, y_val, z_val, y_test, z_test = self.get_y_z_sample(
                y_train, y_val, y_test, z_train, z_val, z_test, p_y_z=p_y_z, p_y=p_y
            )
            print(' (post-sampling) X_test shape, z_test shape, y_test shape:', X_test.shape, z_test.shape, y_test.shape)
            
        
        else:
            # Load multilingual data
            X_train, z_train, y_train = self.load_multilingual('train')
            X_val, z_val, y_val = self.load_multilingual('val')

            # Load test data if needed
            if load_test:
                X_test, z_test, y_test = self.load_multilingual('test')
                
            
         # turn z to one-hot encoding
        if to_one_hot:
            k = len(np.unique(z_train))
            z_train = self.to_one_hot(z_train, num_classes=k)
            z_val = self.to_one_hot(z_val, num_classes=k)
            z_test = self.to_one_hot(z_test, num_classes=k)

            
        # Return data dictionary with same structure as other datasets
        data = {
            'X_train': X_train, 'z_train': z_train, 'y_train': y_train,
            'X_val': X_val, 'z_val': z_val, 'y_val': y_val,
            
        }
        # If embeddings are used, load them
        if load_test:
            data['X_test'] = X_test
            data['z_test'] = z_test
            data['y_test'] = y_test
        
        return data

    def get_group(self, y, z):
        """
        Determine group based on y and z values.
        Args:
            y: label (1 or 0)
            z: language (1, 2, 3)
        """
        # if y == 1 and z == 1
        if y == 1 and z == 1:
            g = 1
        # if y == 1 and z == 2 or 3
        elif y == 1 and (z == 2 or z == 3):
            g = 2
        # if y == 0 and z == 1
        elif y == 0 and z == 1:
            g = 3
        # if y == 0 and z == 0
        elif y == 0 and (z == 2 or z == 3):
            g = 4
        return g
    
    def get_group_granular(self, y, z):
        # if y == 1 and z == 1
        if y == 1 and z == 1:
            g = 1
        # if y == 1 and z == 2
        elif y == 1 and z == 2:
            g = 2
        # if y == 1 and z == 3
        elif y == 1 and z == 3:
            g = 3
        # if y == 0 and z == 1
        elif y == 0 and z == 1:
            g = 4
        # if y == 0 and z == 1
        elif y == 0 and z == 2:
            g = 5
        # if y == 0 and z == 2
        elif y == 0 and z == 3:
            g = 6
        
        return g
    
    def sample_by_probabilities_granular(self, y, z, p_y, p_y_z, n, seed):
        """
        Sample indices to achieve desired probabilities with multi-valued protected attribute
        Specifically designed for multilingual dataset where z has more than 2 values
        
        Args:
            y: binary labels array
            z: multi-valued protected attribute array (e.g., languages)
            p_y: desired P(Y=1)
            p_y_z: desired P(Y=1|Z=1) (not directly used in multi-valued case)
            n: desired total sample size
            seed: random seed
        Returns:
            indices to sample
        """
        # Use get_group_granular for multi-valued z
        get_group_v = np.vectorize(self.get_group_granular)
        
        # Get current group assignments
        groups = get_group_v(y, z)
        
        # Calculate desired counts for each group
        n_y1 = int(n * p_y)
        n_y0 = n - n_y1
        
        # Get unique z values and count them
        langs = len(np.unique(z))
        
        # Sample equally from each language for each y value
        n_y1_per_lang = n_y1 // langs
        n_y0_per_lang = n_y0 // langs
        
        # Sample from each group
        set_seed(seed)
        all_indices = []
        
        # For each group, sample the required number
        for g in range(1, langs*2+1):
            g_idx = np.where(groups == g)[0]
            # Groups 1 to langs are y=1 groups, the rest are y=0 groups
            n_samples = n_y1_per_lang if g <= langs else n_y0_per_lang
            n_samples = min(n_samples, len(g_idx))
            if n_samples > 0:
                sampled = np.random.choice(g_idx, size=n_samples, replace=False)
                all_indices.append(sampled)
        
        # Combine all indices
        indices = np.concatenate(all_indices)
        np.random.shuffle(indices)
        
        return indices

    def get_sample_data(self, X, z, y, n, p_y, p_y_z, multiclass=True, seed=0):
        """
        Sample data from the multilingual dataset according to specific probabilities.
        Args:
            X: text inputs
            z: language attributes
            y: toxicity labels
            n: desired sample size
            p_y: desired P(Y=1)
            p_y_z: desired P(Y=1|Z=1)
            seed: random seed for reproducibility
        Returns:
            Sampled X, z, y
        """

        # turn z to integers if not already
        if not isinstance(z, np.ndarray):
            z = np.array([self.languages_to_int[lang] for lang in z])
        
        # define integer values for the languages we want to sample from
        ints = [self.languages_to_int[lang] for lang in self.languages_for_sample]

        # select only the languages we want to sample from
        mask = np.isin(z, ints)

        # filter the data
        X = [X[i] for i in range(len(X)) if mask[i]]        
        z = z[mask]
        y = y[mask]

        # if necessary, reshape y, z
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)

        # sample indices using the common method
        indices = self.sample_by_probabilities(y, z, p_y, p_y_z, n, seed=seed)
       
        # get the data
        X_sample = [X[i] for i in indices]
        z_sample = z[indices, :]
        y_sample = y[indices, :]

        
        return X_sample, z_sample, y_sample
    




class BiosData(Data):
    def __init__(self):
        super().__init__('bios')
        self.index_y = 21
        self.num_labels = 28

    @property
    def attribute_name(self):
        """BIOS dataset uses gender as the protected attribute"""
        return "gender"
    
    @property
    def target_name(self):
        """BIOS dataset uses profession as the target to preserve"""
        return "profession"

    def get_sample_data(self, X, z, y, n, p_y, p_y_z, seed=0, select_y=True):

        # from the one-hot encoded data y, get the binary y
        y = y[:, self.index_y]

        # sample indices using the common method
        indices = self.sample_by_probabilities(y, z, p_y, p_y_z, n, seed=seed)

        # get the data
        X_sample = X[indices]
        z_sample = z[indices]
        y_sample = y[indices]

        return X_sample, z_sample, y_sample

    def calc_acc_per_prof(self, y_true, y_pred):

        # if y_true is one-hot encoded, turn to single y
        if y_true.shape[1] > 1:
            y_true = y_true.argmax(axis=1)


        # get the unique professions
        professions = np.unique(y_true)

        # go over each profession and calculate the accuracy
        acc_per_prof = {}
        for prof in professions:
            acc = np.mean(y_pred[y_true == prof] == y_true[y_true == prof])
            acc_per_prof[prof] = acc
        
        return acc_per_prof
    
            

    def load_bios(self, split):
        # Load bios data
        with open("data/bios/{}.pickle".format(split), "rb") as f:
            bios_data = pickle.load(f)
        
            X = [d["hard_text_untokenized"] for d in bios_data]
            z = np.array([1 if d["g"]=="f" else 0 for d in bios_data]) # gender labels
            y = np.array([d["p"] for d in bios_data]) # profession labels

        print('n of professions:', len(np.unique(y)))
        print('number of samples:', len(X))
            
        return X, z, y

    


    def load_embeddings(self, split, model_name, embeddings_type='pooler'):
        
        # Load y, z
        with open("data/bios/{}.pickle".format(split), "rb") as f:
             bios_data = pickle.load(f)
             z = np.array([1 if d["g"]=="f" else 0 for d in bios_data]) #
             y = np.array([d["p"] for d in bios_data])
        
        # Load embeddings
        path = "data/embeddings/bios/{}/{}_{}_embeddings.h5".format(model_name, split, embeddings_type)
        print('Loading embeddings from:', path)
        with h5py.File(path, 'r') as f:
            X = f['embeddings'][:]

        
        return X, z, y

    
    def prepare_data(self, load_test=False, embeddings=False, embedding_type='pooler', model_name=None, single_y=True, p_y_z=0.5, p_y=0.5, sample=True, **kwargs):
        """Prepare Bios dataset"""

        # load train and validation data
        if embeddings:
            X_train, z_train, y_train = self.load_embeddings('train', model_name)
            X_val, z_val, y_val = self.load_embeddings('val', model_name)
            
            
        else:
            X_train, z_train, y_train = self.load_bios('train')
            X_val, z_val, y_val = self.load_bios('val')
            

        # get inidces
        prof2ind = {p:i for i,p in enumerate(sorted(set(y_train)))}
        y_train = np.array([prof2ind[p] for p in y_train])
        y_val = np.array([prof2ind[p] for p in y_val])

        # turn to one-hot encoding
        y_train = self.to_one_hot(y_train)
        y_val = self.to_one_hot(y_val)
        
      
        # create a dictionary to store the data
        data = {'X_train': X_train, 'z_train': z_train, 'y_train': y_train,
                'X_val': X_val, 'z_val': z_val, 'y_val': y_val}

        # load test data if needed
        if load_test:
            
            # load test embeddings or bios data
            if embeddings:
                X_test, z_test, y_test = self.load_embeddings('test', model_name)
            else:
                X_test, z_test, y_test = self.load_bios('test')

            # set the y
            y_test = np.array([prof2ind[p] for p in y_test])
            y_test = self.to_one_hot(y_test)

            # add the test data to the dictionary
            data['X_test'] = X_test
            data['z_test'] = z_test
            data['y_test'] = y_test

        if sample:
            # get y, z for the sample
            y_train, z_train, y_val, z_val, y_test, z_test = self.get_y_z_sample(
                y_train, y_val, y_test, z_train, z_val, z_test, p_y_z=p_y_z, p_y=p_y
            )
            # update the data dictionary with the sampled data
            data['y_train'], data['y_val'], data['y_test'] = y_train, y_val, y_test
            data['z_train'], data['z_val'], data['z_test'] = z_train, z_val, z_test
            
            # set the number of labels
            self.num_labels = len(np.unique(y_train))
            
            # if embeddings is false, select the X based on the indices
            if not embeddings:
                data['X_train'] = [data['X_train'][i] for i in self.indices_train]
                data['X_val'] = [data['X_val'][i] for i in self.indices_val]
                data['X_test'] = [data['X_test'][i] for i in self.indices_test]
                
            


        return data

class WaterbirdsData(Data):
    
    def __init__(self):
        super().__init__('waterbirds')
        self.num_labels = 2

    @property
    def attribute_name(self):
        """Waterbirds dataset uses background as the protected attribute"""
        return "place"
    
    @property
    def target_name(self):
        """Waterbirds dataset uses label as the target to preserve"""
        return "y"

    def prepare_data(self, p_y_z,  seed, balance_y=False, **kwargs):
        
        # multiply p_y_z by 100 to match the naming convention if applicable
        if p_y_z < 1:
            p_y_z_int = int(p_y_z * 100)
        else:
            p_y_z_int = p_y_z
        
        # load the embeddings
        X_train, z_train, y_train, X_val, z_val, y_val, X_test, z_test, y_test = self.load_embeddings(
            p_y_z=p_y_z_int, 
            seed=seed
        )
        
        # sample p(y=1) = p_y
        if balance_y:
            
            # determine the n
            n_y_1 = y_train.sum()   
            n_y_0 = len(y_train) - n_y_1
            n_y_min = min(n_y_1, n_y_0)
            n = 2 * n_y_min
            
            # sample indices for training
            indices_train = self.sample_by_probabilities(
                y_train,
                z_train,
                p_y=0.5,
                p_y_z=p_y_z,  # Use the correct parameter value
                n=n,
                seed=seed
            )
            X_train = X_train[indices_train]
            z_train = z_train[indices_train]
            y_train = y_train[indices_train]
            
            # after sampling, print the group counts
            g = np.vectorize(self.get_group)
            groups_train = g(y_train, z_train)
            print('Train group division post sampling:', np.unique(groups_train, return_counts=True))
            
        
        
        return {
            'X_train': X_train, 'z_train': z_train, 'y_train': y_train,
            'X_val': X_val, 'z_val': z_val, 'y_val': y_val,
            'X_test': X_test, 'z_test': z_test, 'y_test': y_test
        }



    def load_embeddings(self,  p_y_z, seed):
        """
        Load the embeddings for the Waterbirds dataset based on correlation and seed.
        Args:  
            corr: correlation type (e.g. 50, 90)
            seed: random seed
        Returns:
            X_train, z_train, y_train: training embeddings and labels
            X_val, z_val, y_val: validation embeddings and labels
            X_test, z_test, y_test: test embeddings and labels
        """
        # Load y, z
        metadata= "data/embeddings/WB/{}/metadata_waterbird_{}_{}_50.csv".format(
            p_y_z, p_y_z, p_y_z
        )
        df = pd.read_csv(metadata)
        df_train = df[df['split'] == 0]
        df_val = df[df['split'] == 1]
        df_test = df[df['split'] == 2]
        
        
        # Extract y and z
        y_train = df_train[self.target_name].values.reshape(-1, 1)
        z_train = df_train[self.attribute_name].values.reshape(-1, 1)   
        y_val = df_val[self.target_name].values.reshape(-1, 1)
        z_val = df_val[self.attribute_name].values.reshape(-1, 1)
        y_test = df_test[self.target_name].values.reshape(-1, 1)
        z_test = df_test[self.attribute_name].values.reshape(-1, 1)
        
       
        # Load embeddings
        embedding_train = 'data/embeddings/WB/{}/waterbird_images_combined_train_{}_embeddings_resnet50_finetuned_settings_kirichenko_seed_{}.pt'.format(p_y_z, p_y_z, seed)
        embedding_val = 'data/embeddings/WB/{}/waterbird_images_combined_val_{}_embeddings_resnet50_finetuned_settings_kirichenko_seed_{}.pt'.format(p_y_z, p_y_z, seed)
        embedding_test = 'data/embeddings/WB/{}/waterbird_images_combined_test_{}_embeddings_resnet50_finetuned_settings_kirichenko_seed_{}.pt'.format(p_y_z, p_y_z, seed)
        
        with open(embedding_train, 'rb') as f:
            print('Loading train embeddings from:', embedding_train)
            X_train = torch.load(embedding_train, weights_only=True, map_location='cpu')
        with open(embedding_val, 'rb') as f:
            print('Loading validation embeddings from:', embedding_val)
            X_val = torch.load(embedding_val, weights_only=True, map_location='cpu')
        with open(embedding_test, 'rb') as f:
            print('Loading test embeddings from:', embedding_test)
            X_test = torch.load(embedding_test, weights_only=True, map_location='cpu')
            
        # Convert to numpy arrays
        X_train,X_val, X_test = X_train.numpy(), X_val.numpy(), X_test.numpy()
        print('Shape of X_train, y_train, z_train:', X_train.shape, y_train.shape, z_train.shape)
        
        # return the embeddings and labels
        return X_train, z_train, y_train, X_val, z_val, y_val, X_test, z_test, y_test
        
        
    
    

class ToyData(Data):

    def __init__(self):
        super().__init__('toy')

    @property
    def attribute_name(self):
        """Toy dataset uses y_c as the protected attribute"""
        return "y_c"
    
    @property
    def target_name(self):
        """Toy dataset uses y_m as the target to preserve"""
        return "y_m"

    def prepare_data(self):
        """Prepare dataset specific data"""
        pass


    def draw_multivariate_normal(self, mu, Sigma, n):
        """
        Draw samples from a multivariate normal distribution.
        """
        return np.random.multivariate_normal(mu, Sigma, n)


    def inv_logit_probability(self, X, v, intercept):
        """
        Calculate the inverse logit probability.
        """
        linear_combination = np.dot(X, v) + intercept
        return 1 / (1 + np.exp(-linear_combination))



    def define_v_unit(self, angle, gamma_m, d, second_m_feature_index, eps = 1e-8, adjustment = True, ):
        """
        Define the unit vector v_m, based on the angle, gamma_m, d, second_m_feature_index, eps, adjustment
        """
    
        # Convert the angle to radians
        angle_rad = math.radians(angle)

        # Calculate the x and y components of the new vector
        v_m_1 = math.cos(angle_rad)
        v_m_2 = math.sin(angle_rad)

        # define an vector of zeros, size d
        v_m = np.zeros(d)

        # define the first and second main feature
        v_m[1] = v_m_1
        v_m[second_m_feature_index] = v_m_2

        # ensure that the vector is a unit vector
        v_m[v_m <= eps] = 0

        # adjust the vector to have the desired gamma_m
        if adjustment:
            if v_m_1 == 0 or v_m_2 == 0:
                v_m = v_m*gamma_m
            else:
                v_m[0] = gamma_m/( 1+ (v_m_1/v_m_2))
                v_m[1] = gamma_m/( 1+ (v_m_2/v_m_1))
        

    
        return v_m

    def generate_X(self, n, d, rho_c_m, rho_c, rho_m, gamma_m, gamma_c, intercept_m, intercept_c, X_variance, angle, n_m_features=1, n_c_features=1):
        """
        Sample X for the Toy dataset, based on the parameters
        """

         # set the number of samples
        self.n = n

        # set the mean for the multivariate normal
        mu = np.zeros(d)
        
        # set the main-task vector  
        v_m = self.define_v_unit(angle, gamma_m, d, second_m_feature_index=0, eps=1e-8, adjustment=True)

        # set the concept vector
        v_c = np.zeros(d)
        v_c[0] = gamma_c

        # set the covariance matrix
        Sigma = np.eye(d) * X_variance

        # set correlation between the main and concept task feature
        Sigma[0, 1] = rho_c_m
        Sigma[1, 0] = rho_c_m
        
        # set correlation between the first column (concept feature) and the last n_c_features columns
        if n_c_features > 1:
            # set correlation between the first column (spurious feature) and the last n_c_features columns
            Sigma[0, 2:(n_c_features+1)] = rho_c
            Sigma[2:(n_c_features+1), 0] = rho_c

            # set correlation between the second column (main feature) and the last n_c_features columns
            Sigma[1, 2:(n_c_features+1)] = rho_c_m
            Sigma[2:(n_c_features+1), 1] = rho_c_m

        # set correlation between the second column (main feature) and the last n_m_features columns
        if n_m_features > 1:
            start = n_c_features + 1
            Sigma[1, start:(start+n_m_features)] = rho_m
            Sigma[start:(start+n_m_features), 1] = rho_m

            # set correlation between the first column (spurious feature) and the last n_m_features columns
            Sigma[0, start:(start+n_m_features)] = rho_c_m
            Sigma[start:(start+n_m_features), 0] = rho_c_m

        # define datapoints X
        X = self.draw_multivariate_normal(mu, Sigma, n)

        return X, v_m, v_c


    def dgp_logit(self, n, d, rho_c_m, rho_c, rho_m, gamma_m, gamma_c, intercept_m, intercept_c, X_variance, angle, n_m_features=1, n_c_features=1):
        """
        Sample data for the Toy dataset, based on the parameters. 
        y_m and y_c are binary labels, v_m and v_c are the main and concept task vectors.
        """
        

        # define datapoints X
        X, v_m, v_c = self.generate_X(n, d, rho_c_m, rho_c, rho_m, gamma_m, gamma_c, intercept_m, intercept_c, X_variance, angle, n_m_features, n_c_features)
        print('v_m:', v_m)
        print('v_c:', v_c)

        # define the concept labels
        p_c = self.inv_logit_probability(X, v_c, intercept_c)
        p_m = self.inv_logit_probability(X, v_m, intercept_m)
        y_c = np.random.binomial(1, p_c).reshape(-1, 1)
        y_m = np.random.binomial(1, p_m).reshape(-1, 1)

        # set the data
        return X, y_m, y_c, v_m, v_c

    def dgp_reg(self, n, d, rho_c_m, rho_c, rho_m, gamma_m, gamma_c, intercept_m, intercept_c, X_variance, angle, n_m_features=1, n_c_features=1):
        """
        Sample data for the Toy dataset, based on the parameters. 
        y_m and y_c are continuous, v_m and v_c are the main and concept task vectors.
        """

        # define datapoints X
        X, v_m, v_c = self.generate_X(n, d, rho_c_m, rho_c, rho_m, gamma_m, gamma_c, intercept_m, intercept_c, X_variance, angle, n_m_features, n_c_features)

        # define the concept labels
        eps_c = np.random.normal(0, 1, n).reshape(-1, 1)
        eps_m = np.random.normal(0, 1, n).reshape(-1, 1)
        beta_c = (v_c).reshape(-1, 1)
        beta_m = (v_m).reshape(-1, 1)

        # define the concept/main labels
        y_c = np.matmul(X, beta_c) + intercept_c + eps_c
        y_m = np.matmul(X, beta_m)  + intercept_m + eps_m

        # set the data
        return X, y_m, y_c, v_m, v_c



