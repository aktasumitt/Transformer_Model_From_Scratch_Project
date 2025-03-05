import nltk
import string
import random
from src.utils import save_as_json
from src.entity.config_entity import DataIngestionConfig

random.seed(42)

nltk.download('punkt')

class DataIngestion():
    
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    def load_data(self):
        
        with open(self.config.data_location_path,"r",encoding="utf-8") as file:
            turkish_sentences=file.readlines()
        return turkish_sentences
    
    def tokenized_data(self,sentence):
            
        sentence=sentence.replace("\n","").lower()
        sentence_tokenized=nltk.word_tokenize(sentence,language="turkish")
        sentence_tokenized=[kelime for kelime in sentence_tokenized if not kelime in string.punctuation+"''"+"``"]
        return sentence_tokenized
    
    def shuffle_tokenized_data(self,sentence_tokenized):

        sentence_tokenized_copy=sentence_tokenized.copy()
        random.shuffle(sentence_tokenized_copy)
            
        return sentence_tokenized_copy
    
    def add_unique_tokens(self,tokenized_sentence:list):
        
        
        tokenized_sentence.insert(0,"<SOS>")
        tokenized_sentence.append("<EOS>")

        tokenized_sentence+=(self.config.max_len_sentence+2-len(tokenized_sentence))*["<PAD>"]
        
        return tokenized_sentence
    
    def initiate_and_save_data(self):
        turkish_data=self.load_data()
        
        tokenized_data_list=[]
        shuffled_tokenize_data_list=[]
        
        for sentence in turkish_data:
            tokenized_sentence=self.tokenized_data(sentence)
            if len(tokenized_sentence)<=self.config.max_len_sentence:
                shuffled_tokenize_sentence=self.shuffle_tokenized_data(tokenized_sentence)
                
                tokenized_sentence_padded=self.add_unique_tokens(tokenized_sentence)
                shuffled_tokenize_sentence_padded=self.add_unique_tokens(shuffled_tokenize_sentence)
                
                tokenized_data_list.append(tokenized_sentence_padded)
                shuffled_tokenize_data_list.append(shuffled_tokenize_sentence_padded)
            
        save_as_json(shuffled_tokenize_data_list,self.config.shuffled_data_path)
        save_as_json(tokenized_data_list,self.config.tokenzied_data_path)
            
        
if __name__=="__main__":
    
    dataingestion=DataIngestion()
    dataingestion.initiate_and_save_data()