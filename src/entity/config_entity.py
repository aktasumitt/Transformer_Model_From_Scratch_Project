from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    
    data_location_path: Path
    tokenzied_data_path: Path
    shuffled_data_path: Path
    max_len_sentence:int
    
@dataclass
class DataTransformationConfig:

    tokenzied_data_path: Path
    shuffled_data_path: Path
    
    transformed_train_dataset: Path
    transformed_test_dataset: Path
    transformed_valid_dataset: Path
    word_box_path: Path
    
    test_split_rate:float
    valid_split_rate:float    
    start_token:str
    stop_token:str
    pad_token:str
    
    
@dataclass
class ModelConfig:
    
  model_save_path: Path
  d_model:int
  dk_model:int
  batch_size:int
  num_token:int
  NX:int
  max_len_sentence:int
  stop_token:str
  device:str
  

@dataclass
class TrainingConfig:

    train_dataset_path: Path
    validation_dataset_path: Path
    model_path: Path
    checkpoint_path: Path
    save_result_path: Path
    final_model_save_path: Path
    batch_size: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    label_smoothing: float
    epochs: int
    device: str
    load_checkpoint: bool

  
@dataclass
class TestConfig:
    final_model_path:Path
    test_dataset_path:Path
    device:str
    batch_size:int
    load_checkpoints_for_test:bool
    save_tested_model:bool
    tested_model_save_path:Path
    test_result_save_path:Path
    best_checkpoints_path:Path
    
    
@dataclass
class PredictionConfig:
    final_model_path:Path
    device:str
    image_size:int
    labels:dict
    predict_data_path:Path
    batch_size:int
    save_prediction_result_path:Path
    
    