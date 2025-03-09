from src.constant.configs import Configs
from src.constant.params import Params
from src.entity.config_entity import (DataIngestionConfig,
                                      DataTransformationConfig,
                                      ModelConfig,
                                      TrainingConfig,
                                      TestConfig,
                                      PredictionConfig)


class Configuration():

    def __init__(self):

        self.config = Configs
        self.params = Params

    def data_ingestion_config(self):

        configuration = DataIngestionConfig(data_location_path=self.config.DATA_LOCATION_PATH,
                                            tokenzied_data_path=self.config.TOKENIZED_DATA_PATH,
                                            shuffled_data_path=self.config.SHUFFLED_TOKENIZED_DATA,
                                            max_len_sentence=self.params.MAX_LEN_SENTENCE)

        return configuration

    def data_transformation_config(self):

        configuration = DataTransformationConfig(tokenzied_data_path=self.config.TOKENIZED_DATA_PATH,
                                                 shuffled_data_path=self.config.SHUFFLED_TOKENIZED_DATA,
                                                 transformed_train_dataset=self.config.TRANSFORMED_TRAIN_DATASET_PATH,
                                                 transformed_test_dataset=self.config.TRANSFORMED_TEST_DATASET_PATH,
                                                 transformed_valid_dataset=self.config.TRANSFORMED_VALID_DATASET_PATH,
                                                 word_box_path=self.config.WORD_BOX_PATH,
                                                 test_split_rate=self.params.TEST_SPLIT_RATE,
                                                 valid_split_rate=self.params.VALID_SPLIT_RATE,
                                                 start_token=self.params.START_TOKEN,
                                                 stop_token=self.params.STOP_TOKEN,
                                                 pad_token=self.params.PAD_TOKEN)

        return configuration

    def model_config(self):

        configuration = ModelConfig(model_save_path=self.config.MODEL_SAVE_PATH,
                                    d_model=self.params.D_MODEL,
                                    dk_model=self.params.DK_MODEL,
                                    batch_size=self.params.BATCH_SIZE,
                                    num_token=self.params.NUM_TOKEN,
                                    NX=self.params.NX,
                                    max_len_sentence=self.params.MAX_LEN_SENTENCE,
                                    stop_token=self.params.STOP_TOKEN_IDX,
                                    device=self.params.DEVICE)

        return configuration

    def training_config(self):

        configuration = TrainingConfig(train_dataset_path=self.config.TRANSFORMED_TRAIN_DATASET_PATH,
                                       validation_dataset_path=self.config.TRANSFORMED_VALID_DATASET_PATH,
                                       model_path=self.config.MODEL_SAVE_PATH,
                                       checkpoint_path=self.config.CHECKPOINT_SAVE_PATH,
                                       save_result_path=self.config.SAVE_TRAINING_RESULT_PATH,
                                       final_model_save_path=self.config.FINAL_MODEL_SAVE_PATH,
                                       batch_size=self.params.BATCH_SIZE,
                                       learning_rate=self.params.LEARNING_RATE,
                                       beta1=self.params.BETA1,
                                       beta2=self.params.BETA2,
                                       epochs=self.params.EPOCHS,
                                       device=self.params.DEVICE,
                                       load_checkpoint=self.params.LOAD_CHECKPOINT_FOR_TRAIN,
                                       epsilon=self.params.epsilon,
                                       label_smoothing=self.params.label_smoothing)

        return configuration

    def test_config(self):

        configuration = TestConfig(final_model_path=self.config.FINAL_MODEL_SAVE_PATH,
                                   test_dataset_path=self.config.TRANSFORMED_TEST_DATASET_PATH,
                                   device=self.params.DEVICE,
                                   batch_size=self.params.BATCH_SIZE,
                                   load_checkpoints_for_test=self.params.LOAD_CHECKPOINT_FOR_TEST,
                                   save_tested_model=self.params.SAVE_TESTED_MODEL,
                                   tested_model_save_path=self.config.TESTED_MODEL_SAVE_PATH,
                                   test_result_save_path=self.config.SAVE_TESTING_RESULT_PATH,
                                   best_checkpoints_path=self.config.BEST_CHECKPOINT_PATH
                                   )

        return configuration

    def prediction_config(self):
        
        configuration = PredictionConfig(final_model_path=self.config.FINAL_MODEL_SAVE_PATH,
                                        device=self.params.DEVICE,
                                        predict_data_path=self.config.PREDICTION_DATA_PATH,
                                        batch_size=self.params.BATCH_SIZE,
                                        save_prediction_result_path=self.config.SAVE_PREDICTION_RESULT_PATH
                                        )

        return configuration

