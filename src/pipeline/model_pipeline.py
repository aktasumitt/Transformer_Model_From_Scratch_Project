from src.config.configuration import Configuration
from src.components.model.transformer_model import model_create_and_save


class ModelPipeline():
    def __init__(self):

        configuration = Configuration()
        self.modelconfig = configuration.model_config()

    def run_model_creating(self):

        model_create_and_save(self.modelconfig)

if __name__=="__main__":
    
    model_pipeline=ModelPipeline()
    model_pipeline.run_model_creating()