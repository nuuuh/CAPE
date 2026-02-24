import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .tsf_dataset import FinetuneDataset, PretrainDataset

import pandas as pd

damaged = ['GIARDIASIS', 
           'MENINGITIS', 
           'BABESIOSIS', 
           'PELLAGRA', 
           'LEPROSY', 
           'ANTHRAX', 
           'DYSENTERY', 
           'ROCKY MOUNTAIN SPOTTED FEVER', 
           'TULAREMIA', 
           'TRICHINIASIS', 
           'BOTULISM', 
           'PSITTACOSIS', 
           'EHRLICHIOSIS/ANAPLASMOSIS', 
           'PNEUMONIA AND INFLUENZA', 
           'TOXIC SHOCK SYNDROME', 
           'SMALLPOX']


cuts = {'HEPATITIS A': 1500,
        'MEASLES': 1600,
        'MUMPS': 500,
        'PERTUSSIS': 800,
        'POLIO': 1500,
        'RUBELLA': 750,
        'SMALLPOX': 600 # 800
        }


np.random.seed(0)

class DataSetWrapper(object):
    def __init__(self, lookback, horizon, batch_size, cut_year, valid_rate, data_path, disease, num_features, max_length, sim=False, ahead=0, shuffle=False, train_rate=0.7):
        self.lookback = lookback
        self.horizon = horizon
        self.sim=sim

        self.batch_size = batch_size
        self.cut_year = cut_year
        self.valid_rate = valid_rate
        self.train_rate = train_rate  # Proportion for training
        self.data_path = data_path
        self.num_features = num_features
        self.max_length = max_length
        self.disease = disease
        try:
            self.total_data = torch.load(data_path)['All']
        except:
            self.total_data = torch.load(data_path)
        
        self.scaler = StandardScaler()
        self.scalers = {}

        self.shuffle=shuffle
        self.ahead=ahead

    def extract(self, disease=None, idx=None, strategy='SUM'):
        # import ipdb; ipdb.set_trace()
        try:
            current_data = self.total_data[disease]
        except:
            current_data = self.total_data[self.disease]
            print(f"Using designated disease: {self.disease}")
        
        if strategy == 'SUM':
            total_infections = {}
            for state, values in current_data.items():
                # import ipdb; ipdb.set_trace()
                infections = values[0][0]
                time = values[0][1]
                for w, i in enumerate(time):
                    i = i.item()
                    if i not in total_infections.keys():
                        total_infections[i] = int(infections.numpy()[w].item())
                    else:
                        total_infections[i] += int(infections.numpy()[w].item())
            
            total_infections = dict(sorted(total_infections.items(), key=lambda x:x[0]))
            # import ipdb; ipdb.set_trace()
            # normalization
            original_infections = np.array([i for i in total_infections.values()])
            norm_infections = (original_infections-original_infections.mean())/original_infections.std()

            time = np.array([i for i in total_infections.keys()])
            infections = np.array([i for i in total_infections.values()])

            # import ipdb; ipdb.set_trace()

            if idx is None:
                if self.cut_year is not None and self.cut_year > 1:
                    # Year-based split (legacy mode)
                    idx=0
                    for i, t in enumerate(time):
                        if int(t/100) >= self.cut_year:
                            idx = i 
                            break
                    # import ipdb; ipdb.set_trace() 
                    # idx = time.index(str(int(self.cut_year)))
                else:
                    # Proportional split
                    train_rate = getattr(self, 'train_rate', 0.7)  # Default to 0.7 if not set
                    idx = int(len(time) * train_rate)

            valid_len = int(len(infections)*self.valid_rate)

            # import ipdb; ipdb.set_trace()

            train_infect = infections[:idx].reshape(-1, 1)
            val_infect = infections[idx:idx+valid_len].reshape(-1, 1)
            test_infect = infections[idx+valid_len:].reshape(-1, 1)

            train_time = time[:idx]
            val_time = time[idx:idx+valid_len]
            test_time = time[idx+valid_len:]

            # Validate sufficient data
            if len(train_infect) == 0:
                raise ValueError(f"No training data available for disease {self.disease}. "
                               f"Check your train_rate or cut_year settings.")
            if len(test_infect) == 0:
                raise ValueError(f"No test data available for disease {self.disease}. "
                               f"Check your train_rate or cut_year settings.")

            # import ipdb; ipdb.set_trace()
            self.scaler.fit(train_infect)
            train_infect = self.scaler.transform(train_infect).reshape(-1)
            if valid_len:
                val_infect = self.scaler.transform(val_infect).reshape(-1)
            test_infect = self.scaler.transform(test_infect).reshape(-1)

            # import ipdb; ipdb.set_trace()
            train_data = self.get_samples(train_infect)
            train_timestamp = self.get_samples(train_time)
            # import ipdb; ipdb.set_trace()
            valid_data = self.get_samples(val_infect)
            valid_timestamp = self.get_samples(val_time)

            test_data = self.get_samples(test_infect)
            test_timestamp = self.get_samples(test_time)
        
        else:
            train_data = []
            valid_data = []
            test_data = []

            train_timestamp = []
            valid_timestamp = []
            test_timestamp = []

            self.state_scalers = {}

            for state, values in current_data.items():
                total_infections = {}
                # import ipdb; ipdb.set_trace()
                infections = values[0][0]
                time = values[0][1]
                for w, i in enumerate(time):
                    i = i.item()
                    if i not in total_infections.keys():
                        total_infections[i] = int(infections.numpy()[w].item())
            
                total_infections = dict(sorted(total_infections.items(), key=lambda x:x[0]))


                time = np.array([i for i in total_infections.keys()])
                infections = np.array([i for i in total_infections.values()])

                idx=None
                if self.cut_year is not None and self.cut_year > 1:
                    # Year-based split (legacy mode)
                    idx=0
                    for i, t in enumerate(time):
                        if int(t/100) >= self.cut_year:
                            idx = i 
                            break
                    # import ipdb; ipdb.set_trace() 
                    # idx = time.index(str(int(self.cut_year)))
                else:
                    # Proportional split
                    train_rate = getattr(self, 'train_rate', 0.7)
                    idx = int(len(time) * train_rate)

                valid_len = int(len(infections)*self.valid_rate)

                # import ipdb; ipdb.set_trace()

                train_infect = infections[:idx].reshape(-1, 1)
                val_infect = infections[idx:idx+valid_len].reshape(-1, 1)
                test_infect = infections[idx+valid_len:].reshape(-1, 1)

                train_time = time[:idx]
                val_time = time[idx:idx+valid_len]
                test_time = time[idx+valid_len:]

                # import ipdb; ipdb.set_trace()
                self.scaler.fit(train_infect)
                self.state_scalers[state] = self.scaler
                
                train_infect = self.scaler.transform(train_infect).reshape(-1)
                if valid_len:
                    val_infect = self.scaler.transform(val_infect).reshape(-1)
                test_infect = self.scaler.transform(test_infect).reshape(-1)

                # import ipdb; ipdb.set_trace()
                train_data += self.get_samples(train_infect)
                train_timestamp += self.get_samples(train_time)
                # import ipdb; ipdb.set_trace()
                valid_data += self.get_samples(val_infect)
                valid_timestamp += self.get_samples(val_time)

                test_data += self.get_samples(test_infect)
                test_timestamp += self.get_samples(test_time)

        return train_timestamp, train_data, valid_timestamp, valid_data, test_timestamp, test_data


    def get_finetune_loaders(self, strategy='SUM'):
        # import ipdb; ipdb.set_trace()
        
        train_timestamp, train_data, valid_timestamp, valid_data, test_timestamp, test_data = self.extract(strategy=strategy)
        self.scalers[self.disease] = self.scaler

        print(f"Training samples: {len(train_data)}")
        print(f"Valid samples: {len(valid_data)}")
        print(f"Testing samples: {len(test_data)}")

        # import ipdb; ipdb.set_trace()

        train_dataset = FinetuneDataset({'data': train_data, "time": train_timestamp}, self.num_features, self.max_length, self.scaler)
        valid_dataset = FinetuneDataset({'data': valid_data, "time": valid_timestamp}, self.num_features, self.max_length, self.scaler)
        test_dataset = FinetuneDataset({'data': test_data, "time": test_timestamp}, self.num_features, self.max_length, self.scaler)
        # import ipdb; ipdb.set_trace()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)

        return train_loader, valid_loader, test_loader
    

    def get_pretrain_loaders(self, strategy='SUM', disease=None):
        train_timestamp = []
        train_data = []
        valid_timestamp = []
        valid_data = []
        test_timestamp = []
        test_data = []

        jumped = 0

        if disease is not None:
            idx=None
            tmp_train_timestamp, tmp_train_data, tmp_valid_timestamp, tmp_valid_data, tmp_test_timestamp, tmp_test_data = self.extract(disease, idx, strategy)
            try:
                self.scalers[disease] = self.state_scalers
            except:
                self.scalers[disease] = self.scaler

            train_timestamp += tmp_train_timestamp
            train_data += tmp_train_data
            valid_timestamp += tmp_valid_timestamp
            valid_data += tmp_valid_data
            test_timestamp += tmp_test_timestamp
            test_data += tmp_test_data
        else:
            for disease in self.total_data.keys():
                # import ipdb; ipdb.set_trace()
                idx=None
                # if strategy =='SUM':
                #     if disease in damaged:
                #         # print(disease)
                #         jumped += 1
                #         continue
                #     # import ipdb; ipdb.set_trace()
                #     # assert (disease in cuts) == True
                #     try:
                #         idx = cuts[disease]
                #     except:
                #         pass
                # idx=None
                tmp_train_timestamp, tmp_train_data, tmp_valid_timestamp, tmp_valid_data, tmp_test_timestamp, tmp_test_data = self.extract(disease, idx, strategy)
                try:
                    self.scalers[disease] = self.state_scalers
                except:
                    self.scalers[disease] = self.scaler

                # import ipdb; ipdn.set_trace()

                train_timestamp += tmp_train_timestamp
                train_data += tmp_train_data
                valid_timestamp += tmp_valid_timestamp
                valid_data += tmp_valid_data
                test_timestamp += tmp_test_timestamp
                test_data += tmp_test_data
        

        print(f"Total Training samples: {len(train_data)}")
        print(f"Total Valid samples: {len(valid_data)}")
        print(f"Total Testing samples: {len(test_data)}")
        # data_size.to_excel("pretrain_datasets.xlsx", index=False) 
        train_dataset = PretrainDataset({'data': train_data, "time": train_timestamp}, self.num_features, self.max_length)
        valid_dataset = PretrainDataset({'data': valid_data, "time": valid_timestamp}, self.num_features, self.max_length)
        test_dataset = PretrainDataset({'data': test_data, "time": test_timestamp}, self.num_features, self.max_length)
        # import ipdb; ipdb.set_trace()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)
        if len(valid_dataset.data)!=0:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)
        else:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        return train_loader, valid_loader, test_loader

    def get_samples(self, data):
        tmp = []
        for i in range(len(data) - self.lookback - self.horizon - self.ahead):
            history = data[i : i+self.lookback]
            future = data[i+self.lookback+self.ahead : i+self.lookback+self.ahead+self.horizon]
            tmp.append((history, future))
        return tmp
    
    # def get_pretrain_samples(self, data):
    #     tmp = []
    #     for i in range(len(data) - self.lookback):
    #         history = data[i : i+self.lookback]
    #         tmp.append(history)
    #     return tmp
    
    def inverse_transform(self, disease, data):
        return self.scalers[disease].inverse_transform(data)
    
    def get_next_token_pretrain_loaders(self, strategy='SUM', token_size=4):
        """
        Get dataloaders for next-token-prediction pretraining
        
        For pretraining (similar to language models):
        - Use ALL diseases from the dataset (tycho_US.pt recommended)
        - strategy='SUM': Aggregate all regional series into one national series per disease (faster)
        - strategy='ALL': Use each regional series individually (more data, slower)
        - This allows the model to learn diverse epidemic patterns
        """
        from .tsf_dataset import NextTokenPretrainDataset
        
        train_data = []
        valid_data = []
        test_data = []

        train_timestamp = []
        valid_timestamp = []
        test_timestamp = []

        diseases_processed = 0
        diseases_skipped = 0
        total_series = 0
        
        # ALWAYS use all diseases for pretraining (like LM pretraining on all documents)
        all_diseases = list(self.total_data.keys())
        print(f"[NextToken Pretraining] Loading all {len(all_diseases)} diseases (strategy={strategy})...")
        
        for disease in tqdm(all_diseases, desc="Loading diseases", unit="disease"):
            try:
                current_disease_data = self.total_data[disease]
                
                # Check if this disease has regional data (dict of states) or single series (Tensor)
                if isinstance(current_disease_data, dict) and strategy == 'SUM':
                    # Aggregate all states into one national time series per disease
                    total_infections = {}
                    for state, values in current_disease_data.items():
                        try:
                            infections = values[0][0].numpy()
                            time = values[0][1].numpy()
                            for w, t in enumerate(time):
                                t_val = int(t.item()) if hasattr(t, 'item') else int(t)
                                if t_val not in total_infections:
                                    total_infections[t_val] = int(infections[w])
                                else:
                                    total_infections[t_val] += int(infections[w])
                        except:
                            continue
                    
                    if len(total_infections) == 0:
                        diseases_skipped += 1
                        continue
                    
                    # Sort by time
                    total_infections = dict(sorted(total_infections.items()))
                    time = np.array(list(total_infections.keys()))
                    infections = np.array(list(total_infections.values()), dtype=np.float32)
                    
                    # Split and normalize
                    total_len = len(infections)
                    train_end = int(total_len * 0.7)
                    valid_end = int(total_len * 0.85)
                    
                    train_infect = infections[:train_end].reshape(-1, 1)
                    valid_infect = infections[train_end:valid_end].reshape(-1, 1)
                    test_infect = infections[valid_end:].reshape(-1, 1)
                    
                    train_time = time[:train_end]
                    valid_time = time[train_end:valid_end]
                    test_time = time[valid_end:]
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    
                    if len(train_infect) > 0:
                        scaler.fit(train_infect)
                        train_infect = scaler.transform(train_infect).reshape(-1)
                        if len(valid_infect) > 0:
                            valid_infect = scaler.transform(valid_infect).reshape(-1)
                        if len(test_infect) > 0:
                            test_infect = scaler.transform(test_infect).reshape(-1)
                        
                        train_data += self.get_samples(train_infect)
                        train_timestamp += self.get_samples(train_time)
                        valid_data += self.get_samples(valid_infect)
                        valid_timestamp += self.get_samples(valid_time)
                        test_data += self.get_samples(test_infect)
                        test_timestamp += self.get_samples(test_time)
                        
                        total_series += 1
                    
                    diseases_processed += 1
                    tqdm.write(f"✓ {disease} (aggregated, {total_len} points)")
                
                elif isinstance(current_disease_data, dict):
                    # tycho_US.pt format: dict of states, each with time series
                    # Treat each state's time series as a separate "document"
                    series_in_disease = 0
                    for state, values in current_disease_data.items():
                        try:
                            infections = values[0][0].numpy()
                            time = values[0][1].numpy()
                            
                            # Use entire time series (no cut_year split for pretraining)
                            # Split into train/valid/test by time to maintain temporal order
                            total_len = len(infections)
                            train_end = int(total_len * 0.7)
                            valid_end = int(total_len * 0.85)
                            
                            train_infect = infections[:train_end].reshape(-1, 1)
                            valid_infect = infections[train_end:valid_end].reshape(-1, 1)
                            test_infect = infections[valid_end:].reshape(-1, 1)
                            
                            train_time = time[:train_end]
                            valid_time = time[train_end:valid_end]
                            test_time = time[valid_end:]
                            
                            # Normalize with separate scaler per series
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            
                            if len(train_infect) > 0:
                                scaler.fit(train_infect)
                                train_infect = scaler.transform(train_infect).reshape(-1)
                                
                                if len(valid_infect) > 0:
                                    valid_infect = scaler.transform(valid_infect).reshape(-1)
                                if len(test_infect) > 0:
                                    test_infect = scaler.transform(test_infect).reshape(-1)
                                
                                # Create samples from each series
                                train_data += self.get_samples(train_infect)
                                train_timestamp += self.get_samples(train_time)
                                valid_data += self.get_samples(valid_infect)
                                valid_timestamp += self.get_samples(valid_time)
                                test_data += self.get_samples(test_infect)
                                test_timestamp += self.get_samples(test_time)
                                
                                total_series += 1
                                series_in_disease += 1
                        except Exception as e:
                            continue
                    
                    diseases_processed += 1
                    print(f"✓ ({series_in_disease} series)")
                    
                elif isinstance(current_disease_data, torch.Tensor):
                    # Simple tensor format (like total.pt)
                    infections = current_disease_data.numpy()
                    time = np.arange(len(infections))
                    
                    # Use entire time series
                    total_len = len(infections)
                    train_end = int(total_len * 0.7)
                    valid_end = int(total_len * 0.85)
                    
                    train_infect = infections[:train_end].reshape(-1, 1)
                    valid_infect = infections[train_end:valid_end].reshape(-1, 1)
                    test_infect = infections[valid_end:].reshape(-1, 1)
                    
                    train_time = time[:train_end]
                    valid_time = time[train_end:valid_end]
                    test_time = time[valid_end:]
                    
                    # Normalize
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    
                    if len(train_infect) > 0:
                        scaler.fit(train_infect)
                        train_infect = scaler.transform(train_infect).reshape(-1)
                        
                        if len(valid_infect) > 0:
                            valid_infect = scaler.transform(valid_infect).reshape(-1)
                        if len(test_infect) > 0:
                            test_infect = scaler.transform(test_infect).reshape(-1)
                        
                        train_data += self.get_samples(train_infect)
                        train_timestamp += self.get_samples(train_time)
                        valid_data += self.get_samples(valid_infect)
                        valid_timestamp += self.get_samples(valid_time)
                        test_data += self.get_samples(test_infect)
                        test_timestamp += self.get_samples(test_time)
                        
                        total_series += 1
                    
                    diseases_processed += 1
                
            except Exception as e:
                diseases_skipped += 1
                tqdm.write(f"  ✗ Skipped {disease}: {str(e)[:50]}")
                continue
        
        print(f"\n[NextToken Pretraining Summary]")
        print(f"  Diseases processed: {diseases_processed}")
        print(f"  Total time series: {total_series}")
        print(f"  Diseases skipped: {diseases_skipped}")
        print(f"[NextToken] Total Training samples: {len(train_data)}")
        print(f"[NextToken] Total Valid samples: {len(valid_data)}")
        print(f"[NextToken] Total Testing samples: {len(test_data)}")
        
        train_dataset = NextTokenPretrainDataset({'data': train_data, "time": train_timestamp}, 
                                                   self.num_features, self.max_length, token_size=token_size)
        valid_dataset = NextTokenPretrainDataset({'data': valid_data, "time": valid_timestamp}, 
                                                   self.num_features, self.max_length, token_size=token_size)
        test_dataset = NextTokenPretrainDataset({'data': test_data, "time": test_timestamp}, 
                                                  self.num_features, self.max_length, token_size=token_size)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)
        if len(valid_dataset.valid_indices) != 0:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)
        else:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        return train_loader, valid_loader, test_loader
    
    def get_next_token_finetune_loaders(self, token_size=4):
        """Get dataloaders for next-token-prediction finetuning"""
        from .tsf_dataset import NextTokenFinetuneDataset
        
        if self.disease != 'All':
            train_timestamp, train_data, valid_timestamp, valid_data, test_timestamp, test_data = self.extract(self.disease)
            scalar = self.scaler
        else:
            raise ValueError("Next-token finetuning requires a specific disease, not 'All'")
        
        print(f"[NextToken Finetune] Total Training samples: {len(train_data)}")
        print(f"[NextToken Finetune] Total Valid samples: {len(valid_data)}")
        print(f"[NextToken Finetune] Total Testing samples: {len(test_data)}")
        
        train_dataset = NextTokenFinetuneDataset({'data': train_data, "time": train_timestamp}, 
                                                   self.num_features, self.max_length, 
                                                   token_size=token_size, scalar=scalar)
        valid_dataset = NextTokenFinetuneDataset({'data': valid_data, "time": valid_timestamp}, 
                                                   self.num_features, self.max_length, 
                                                   token_size=token_size, scalar=scalar)
        test_dataset = NextTokenFinetuneDataset({'data': test_data, "time": test_timestamp}, 
                                                  self.num_features, self.max_length, 
                                                  token_size=token_size, scalar=scalar)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        if len(valid_dataset.valid_indices) != 0:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        else:
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False)

        return train_loader, valid_loader, test_loader
    
    def get_synthetic_next_token_loaders(
        self, 
        token_size: int = 4,
        num_train: int = 1000,
        num_valid: int = 200,
        num_test: int = 200,
        univariate: bool = False,
        use_groups: bool = True,
        group_ratio: float = 0.5,
        min_compartments: int = 3,
        max_compartments: int = 7,
        min_transitions: int = 3,
        max_transitions: int = 8,
        min_weeks: int = 52,
        max_weeks: int = 260,
        time_resolution: str = 'weekly',
        daily_ratio: float = 0.0,
        rng_seed: int = None,
        streaming: bool = True,
        num_workers: int = 4,
        # NEW: GP augmentation parameters
        use_gp_augmentation: bool = False,
        gp_ratio: float = 0.2
    ):
        """
        Get dataloaders for next-token-prediction using SYNTHETIC data generated on-the-fly
        
        Uses epirecipe pipeline to generate diverse epidemic models with random:
        - Compartmental structures (SIR, SEIR, SEIRS, etc.)
        - Transition dynamics
        - Parameters (R0, population, etc.)
        - Seasonal forcing patterns (NEW)
        
        NEW: Supports GP augmentation for learning periodic patterns (KernelSynth-inspired)
        
        Args:
            token_size: Number of time steps per token
            num_train: Number of synthetic epidemics for training
            num_valid: Number of synthetic epidemics for validation
            num_test: Number of synthetic epidemics for testing
            univariate: If True, use only infected (I) compartment. If False, use all compartments
            min_compartments: Minimum compartments in generated models
            max_compartments: Maximum compartments in generated models
            min_transitions: Minimum transition rules
            max_transitions: Maximum transition rules
            min_weeks: Minimum simulation duration
            max_weeks: Maximum simulation duration
            time_resolution: 'weekly', 'daily', or 'mixed' for mixed weekly/daily training
            daily_ratio: When time_resolution='mixed', fraction of samples that are daily (0.0-1.0)
            rng_seed: Random seed for reproducibility
            streaming: Use streaming dataset (infinite) or pre-generated (fixed)
            num_workers: Number of DataLoader workers for parallel data loading (0=single-threaded)
            use_gp_augmentation: Whether to mix in GP-generated samples for periodic patterns
            gp_ratio: Fraction of samples from GP generator (0.0-1.0), default 0.2 (20%)
            
        Returns:
            train_loader, valid_loader, test_loader
        """
        from .streaming_synthetic_dataset import (
            StreamingSyntheticDataset, 
            FixedSyntheticDataset,
            collate_variable_length
        )
        
        print("="*80)
        print("SYNTHETIC DATA GENERATION FOR NEXT-TOKEN PRETRAINING")
        print("="*80)
        print(f"Mode: {'STREAMING' if streaming else 'PRE-GENERATED'}")
        print(f"Data type: {'Univariate (I only)' if univariate else 'Multivariate (all compartments)'}")
        print(f"Group-stratified: {'Yes' if use_groups else 'No'} (ratio: {group_ratio:.0%})")
        print(f"GP Augmentation: {'Yes' if use_gp_augmentation else 'No'} (ratio: {gp_ratio:.0%})")
        print(f"Train epidemics: {num_train}")
        print(f"Valid epidemics: {num_valid}")
        print(f"Test epidemics: {num_test}")
        print(f"Compartments range: [{min_compartments}, {max_compartments}]")
        print(f"Transitions range: [{min_transitions}, {max_transitions}]")
        print(f"Simulation weeks: [{min_weeks}, {max_weeks}]")
        print("="*80)
        
        if streaming:
            # Streaming mode: generate data on-the-fly during training
            print("\n[Streaming Mode] Data will be generated on-the-fly during training")
            print("  ✓ No upfront generation time")
            print("  ✓ Infinite data diversity (different each epoch)")
            print("  ✓ Memory efficient")
            if use_groups:
                print(f"  ✓ Group-stratified models ({group_ratio:.0%} of samples)")
            print()
            
            # Training: streaming dataset
            train_dataset = StreamingSyntheticDataset(
                num_samples_per_epoch=num_train,
                token_size=token_size,
                univariate=univariate,
                use_groups=use_groups,
                group_ratio=group_ratio,
                min_compartments=min_compartments,
                max_compartments=max_compartments,
                min_transitions=min_transitions,
                max_transitions=max_transitions,
                min_weeks=min_weeks,
                max_weeks=max_weeks,
                time_resolution=time_resolution,
                daily_ratio=daily_ratio,
                rng_seed=rng_seed,
                use_gp_augmentation=use_gp_augmentation,
                gp_ratio=gp_ratio
            )
            
            # Validation/Test: fixed datasets for consistent evaluation (no GP augmentation)
            print("Generating fixed validation set...")
            valid_dataset = FixedSyntheticDataset(
                num_samples=num_valid,
                token_size=token_size,
                univariate=univariate,
                use_groups=use_groups,
                group_ratio=group_ratio,
                min_compartments=min_compartments,
                max_compartments=max_compartments,
                min_transitions=min_transitions,
                max_transitions=max_transitions,
                min_weeks=min_weeks,
                max_weeks=max_weeks,
                time_resolution=time_resolution,
                daily_ratio=daily_ratio,
                rng_seed=rng_seed + 1000 if rng_seed else None
            )
            
            print("Generating fixed test set...")
            test_dataset = FixedSyntheticDataset(
                num_samples=num_test,
                token_size=token_size,
                univariate=univariate,
                use_groups=use_groups,
                group_ratio=group_ratio,
                min_compartments=min_compartments,
                max_compartments=max_compartments,
                min_transitions=min_transitions,
                max_transitions=max_transitions,
                min_weeks=min_weeks,
                max_weeks=max_weeks,
                time_resolution=time_resolution,
                daily_ratio=daily_ratio,
                rng_seed=rng_seed + 2000 if rng_seed else None
            )
            
            print(f"\n[Streaming] Training: ~{len(train_dataset)} samples per epoch (generated on-the-fly)")
            print(f"[Fixed] Validation: {len(valid_dataset)} samples")
            print(f"[Fixed] Test: {len(test_dataset)} samples")
            
            # Create dataloaders with parallel workers for faster data generation
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                collate_fn=collate_variable_length,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None
            )
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=collate_variable_length,
                num_workers=min(num_workers, 2),  # Fewer workers for validation
                persistent_workers=(num_workers > 0)
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=collate_variable_length,
                num_workers=0  # Single-threaded for test
            )
            
            return train_loader, valid_loader, test_loader
        
        # Non-streaming mode: pre-generate all data (fallback for compatibility)
        print("\n[Pre-generation Mode] Generating all data upfront...")
        print("  Note: For large datasets, use --synthetic_streaming True instead")
        print()
        
        from .synthetic_data_generator import SyntheticEpidemicGenerator
        from sklearn.preprocessing import StandardScaler
        
        # Initialize generator
        generator = SyntheticEpidemicGenerator(
            min_compartments=min_compartments,
            max_compartments=max_compartments,
            min_transitions=min_transitions,
            max_transitions=max_transitions,
            min_weeks=min_weeks,
            max_weeks=max_weeks,
            rng_seed=rng_seed
        )
        
        # Generate train data
        print("\nGenerating TRAINING data...")
        train_samples = generator.generate_batch(num_train, univariate=univariate, verbose=True)
        
        # Generate validation data
        print("\nGenerating VALIDATION data...")
        valid_samples = generator.generate_batch(num_valid, univariate=univariate, verbose=True)
        
        # Generate test data
        print("\nGenerating TEST data...")
        test_samples = generator.generate_batch(num_test, univariate=univariate, verbose=True)
        
        # Process into dataset format
        def process_samples(samples):
            data_list = []
            time_list = []
            
            for data, time, compartments, R0 in samples:
                # data shape: [num_weeks, num_features]
                num_features = data.shape[1]
                
                # Handle each feature separately to maintain temporal structure
                for feat_idx in range(num_features):
                    feat_data = data[:, feat_idx]
                    
                    # Split into train/valid/test temporally
                    total_len = len(feat_data)
                    train_end = int(total_len * 0.7)
                    valid_end = int(total_len * 0.85)
                    
                    train_part = feat_data[:train_end].reshape(-1, 1)
                    valid_part = feat_data[train_end:valid_end].reshape(-1, 1)
                    test_part = feat_data[valid_end:].reshape(-1, 1)
                    
                    train_time = time[:train_end]
                    valid_time = time[train_end:valid_end]
                    test_time = time[valid_end:]
                    
                    # Normalize each series independently
                    scaler = StandardScaler()
                    if len(train_part) > 0:
                        scaler.fit(train_part)
                        train_part = scaler.transform(train_part).reshape(-1)
                        if len(valid_part) > 0:
                            valid_part = scaler.transform(valid_part).reshape(-1)
                        if len(test_part) > 0:
                            test_part = scaler.transform(test_part).reshape(-1)
                        
                        # Create samples using get_samples logic
                        train_samples_feat = self.get_samples(train_part)
                        valid_samples_feat = self.get_samples(valid_part)
                        test_samples_feat = self.get_samples(test_part)
                        
                        train_time_samples = self.get_samples(train_time)
                        valid_time_samples = self.get_samples(valid_time)
                        test_time_samples = self.get_samples(test_time)
                        
                        return {
                            'train_data': train_samples_feat,
                            'train_time': train_time_samples,
                            'valid_data': valid_samples_feat,
                            'valid_time': valid_time_samples,
                            'test_data': test_samples_feat,
                            'test_time': test_time_samples
                        }
            
            return None
        
        # Collect all samples
        train_data, valid_data, test_data = [], [], []
        train_time, valid_time, test_time = [], [], []
        
        # Helper function to process samples
        # For synthetic data in next-token prediction: Just prepare raw sequences for tokenization
        # No need for lookback/horizon windows - that's for traditional forecasting
        def process_sample_list(samples, split_name):
            local_data = []
            local_time = []
            
            for data, time, compartments in tqdm(samples, desc=f"Processing {split_name}", unit="series"):
                num_features = data.shape[1]
                for feat_idx in range(num_features):
                    feat_data = data[:, feat_idx].reshape(-1, 1)
                    feat_time = time
                    
                    # Normalize the entire series
                    scaler = StandardScaler()
                    if len(feat_data) > 0:
                        scaler.fit(feat_data)
                        feat_data = scaler.transform(feat_data).reshape(-1)
                        
                        # For next-token prediction, just store the entire normalized sequence
                        # The NextTokenPretrainDataset will handle tokenization
                        # No need for sliding windows with lookback/horizon
                        total_len = len(feat_data)
                        min_len = token_size * 2  # Need at least 2 tokens
                        
                        if total_len >= min_len:
                            # Store as a single sample: (full_sequence, empty_future)
                            local_data.append((feat_data, np.array([])))
                            local_time.append((feat_time, np.array([])))
            
            return local_data, local_time
        
        print("\nProcessing and normalizing generated data...")
        
        # Process each split separately - each uses different epidemic samples
        train_data, train_time = process_sample_list(train_samples, "train")
        valid_data, valid_time = process_sample_list(valid_samples, "valid")
        test_data, test_time = process_sample_list(test_samples, "test")
        
        print(f"\n[Pre-generated] Training samples: {len(train_data)}")
        print(f"\n[Pre-generated] Validation samples: {len(valid_data)}")
        print(f"\n[Pre-generated] Test samples: {len(test_data)}")
        
        # Create datasets using the old approach (for backward compatibility)
        from .tsf_dataset import NextTokenPretrainDataset
        
        train_dataset = NextTokenPretrainDataset(
            {'data': train_data, "time": train_time}, 
            self.num_features, self.max_length, token_size=token_size
        )
        valid_dataset = NextTokenPretrainDataset(
            {'data': valid_data, "time": valid_time}, 
            self.num_features, self.max_length, token_size=token_size
        )
        test_dataset = NextTokenPretrainDataset(
            {'data': test_data, "time": test_time}, 
            self.num_features, self.max_length, token_size=token_size
        )
        
        # Custom collate function to handle variable-length sequences
        def collate_variable_length(batch):
            """Pad sequences to the same length within a batch"""
            # Find max sequence length in this batch
            max_len = max(item['input'].shape[0] for item in batch)
            
            # Pad each sequence
            padded_inputs = []
            padded_targets = []
            padded_times = []
            
            for item in batch:
                seq_len = item['input'].shape[0]
                token_size = item['input'].shape[1]
                
                if seq_len < max_len:
                    # Pad with zeros
                    pad_len = max_len - seq_len
                    input_pad = torch.zeros(pad_len, token_size)
                    target_pad = torch.zeros(pad_len, token_size)
                    time_pad = torch.zeros(pad_len, token_size)
                    
                    padded_input = torch.cat([item['input'], input_pad], dim=0)
                    padded_target = torch.cat([item['target'], target_pad], dim=0)
                    padded_time = torch.cat([item['input_time'], time_pad], dim=0)
                else:
                    padded_input = item['input']
                    padded_target = item['target']
                    padded_time = item['input_time']
                
                padded_inputs.append(padded_input)
                padded_targets.append(padded_target)
                padded_times.append(padded_time)
            
            return {
                'input': torch.stack(padded_inputs),
                'target': torch.stack(padded_targets),
                'input_time': torch.stack(padded_times)
            }
        
        # Create dataloaders with custom collate function for variable lengths
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=False, 
                                 shuffle=self.shuffle, collate_fn=collate_variable_length)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=False, 
                                 shuffle=False, collate_fn=collate_variable_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, drop_last=False, 
                                 shuffle=False, collate_fn=collate_variable_length)
        
        return train_loader, valid_loader, test_loader

    



    


