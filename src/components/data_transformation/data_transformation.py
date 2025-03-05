import torch
from torch.utils.data import TensorDataset,random_split
from src.utils import load_json,save_obj,save_as_json
from src.entity.config_entity import DataTransformationConfig

class DataTransformation():
    def __init__(self,config:DataTransformationConfig):
        
        self.config=config
        self.pad_token=config.pad_token
        self.start_token=config.start_token
        self.stop_token=config.stop_token

    def create_word_box(self,data):

        word_box={self.pad_token:0,
                  self.start_token:1,
                  self.stop_token:2}
        
        word_set=set([self.pad_token,self.start_token,self.stop_token])
        idx=len(word_box)
        
        for i,sentence in enumerate(data):
            for kelime in sentence:
                if not kelime in word_set:
                    word_box[kelime]=idx
                    word_set.add(kelime)
                    idx+=1
        return word_box

    def transform_word_to_idx(self,data,word_box):
        
        sentence_idx_list=[]

        for i,cumle in enumerate(data):
            cumle_idx_list=[]
            for kelime in cumle:
                if kelime in word_box:
                    cumle_idx_list.append(word_box[kelime])
                else:
                    print(f"{kelime} couldn't find")
                    print(cumle)
                    print(i)

            sentence_idx_list.append(torch.tensor(cumle_idx_list))
        
        return torch.stack(sentence_idx_list)
    
    def create_torch_dataset(self,input,output):
        return TensorDataset(input,output)
    
    def random_split_dataset(self,dataset):
        valid_len=int(len(dataset)*self.config.valid_split_rate)
        test_len=int(len(dataset)*self.config.test_split_rate)
        train_len=len(dataset)-valid_len-test_len
        
        train,valid,test=random_split(dataset,[train_len,valid_len,test_len])
        return train,valid,test
        
    def initiate_data_transformation(self):
        tokenized_data=load_json(self.config.tokenzied_data_path)
        shuffled_tokenized_data=load_json(self.config.shuffled_data_path)
        
        word_box=self.create_word_box(tokenized_data,max_sentence_len=15)
        save_as_json(data=word_box,save_path=self.config.word_box_path)
        
        tokenized_idx_data=self.transform_word_to_idx(tokenized_data,word_box)
        tokenized_shuffled_idx_data=self.transform_word_to_idx(shuffled_tokenized_data,word_box)
        
        tensor_dataset=self.create_torch_dataset(tokenized_idx_data,tokenized_shuffled_idx_data)
        
        train_dataset,valid_dataset,test_dataset=self.random_split_dataset(tensor_dataset)
        save_obj(train_dataset,self.config.transformed_train_dataset)
        save_obj(valid_dataset,self.config.transformed_valid_dataset)
        save_obj(test_dataset,self.config.transformed_test_dataset)
        
    
if __name__=="__main__":
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation()
        
        
    
