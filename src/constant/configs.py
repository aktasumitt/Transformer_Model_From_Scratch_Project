class Configs():
    
    # After data ingestion
    DATA_LOCATION_PATH = "local_data/wiki.tr.txt"
    TOKENIZED_DATA_PATH = "artifacts/data_ingestion/tokenized_data.json"
    SHUFFLED_TOKENIZED_DATA="artifacts/data_ingestion/shuffled_tokenized_data.json"

    # After transformation
    TRANSFORMED_TRAIN_DATASET_PATH = "artifacts/data_transformation/train_dataset.pth"
    TRANSFORMED_TEST_DATASET_PATH = "artifacts/data_transformation/test_dataset.pth"
    TRANSFORMED_VALID_DATASET_PATH = "artifacts/data_transformation/valid_dataset.pth"
    WORD_BOX_PATH = "artifacts/data_transformation/word_box.json"

    # After creating model
    MODEL_SAVE_PATH = "artifacts/model/transformer_model.pth"

    # After training
    CHECKPOINT_SAVE_PATH = "callbacks/checkpoints/checkpoint_last.pth.tar"
    SAVE_TRAINING_RESULT_PATH = "results/train_results.json"
    FINAL_MODEL_SAVE_PATH = "callbacks/final_model/final_model.pth"
    
    # After Testing
    TESTED_MODEL_SAVE_PATH = "callbacks/tested_model/tested_best_model.pth"
    SAVE_TESTING_RESULT_PATH = "results/test_results.json"
    BEST_CHECKPOINT_PATH = "callbacks/checkpoints/checkpoint_5-epoch.pth.tar"  # change this as your results
    
    # After prediction
    SAVE_PREDICTION_RESULT_PATH = "predict_artifact/results/result.json"
    PREDICTION_DATA_PATH= "predict_artifact/images"
    
    