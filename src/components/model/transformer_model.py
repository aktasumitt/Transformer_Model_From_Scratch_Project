import torch.nn as nn
import torch
from src.components.model.encoder import Encoder
from src.components.model.decoder import Decoder
from src.components.model.layers import Embedding ,PositionalEncoding
from src.utils import save_obj
from src.entity.config_entity import ModelConfig


class Transformer(nn.Module):
    def __init__(self,config:ModelConfig):
        super(Transformer, self).__init__()
        self.config=config
        self.batch_size = config.batch_size
        self.Nx = config.NX
        self.max_len = config.max_len_sentence
        self.stop_token = config.stop_token
        self.d_model = config.d_model
        self.device= config.device
        self.num_token=config.num_token
        self.dk_model=config.dk_model
        
        self.embedding = Embedding(self.d_model, token_size=self.num_token, pad_idx=0)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_len, self.device)
        self.encoder = nn.ModuleList([Encoder(self.d_model, self.dk_model, self.batch_size, self.max_len) for _ in range(self.Nx)])
        self.decoder = nn.ModuleList([Decoder(self.d_model, self.dk_model, self.batch_size, self.max_len) for _ in range(self.Nx)])
        
        self.flatten = nn.Flatten(-2, -1)
        self.last_linear = nn.Linear(self.d_model * self.max_len, self.num_token)
    
    def Encoder_Stack(self, input):
        # ENCODER
        embed_enc = self.embedding(input)  # same embedding with encoder and scale with root(d_model)
        encoder_in = self.positional_encoding(embed_enc)
        
        for i in range(self.Nx):  # Stack Encoder
            encoder_in = self.encoder[i](encoder_in)
        
        return encoder_in
    
    def Decoder_Stack(self, input_decoder, encoder_out):
        embed_dec = self.embedding(input_decoder)  
        decoder_in = self.positional_encoding(embed_dec)
        
        for i in range(self.Nx):
            decoder_in = self.decoder[i](decoder_in, encoder_out)        
        
        return decoder_in
        
    def forward(self, input_encoder, input_decoder, targets_decoder=None):
        stop_token_idx = 0
        output_list = []
        
        # ENCODER
        encoder_out = self.Encoder_Stack(input_encoder)
        
        # DECODER
        for i in range(self.max_len - 1):
            decoder_out = self.Decoder_Stack(input_decoder, encoder_out)
            
            # Prediction layer
            flat_out = self.flatten(decoder_out)
            out_transformer = self.last_linear(flat_out)
            output_list.append(out_transformer)
            
            if targets_decoder is not None:  # For training and we can use Teacher Forcing method
                stop_token_idx += (targets_decoder[:, i] == self.stop_token).sum().item()
                if stop_token_idx == self.batch_size:
                    break
                else:
                    
                    input_decoder = input_decoder.clone()
                    input_decoder[:, i+1] = targets_decoder[:, i]
            else:
                soft = nn.functional.softmax(out_transformer, dim=-1)
                _, pred = torch.max(soft, -1)
                
                if pred.item() == self.stop_token:
                    break
                else: 
                    input_decoder = input_decoder.clone()
                    input_decoder[:, i+1] = pred.item()
        
        return torch.stack(output_list).permute(1, 0, 2)  # (max_len, B, num_tokens) --> (B, max_len, num_tokens)
    
    

def model_create_and_save(config:ModelConfig):
    model=Transformer(config)
    save_obj(data=model,save_path=config.model_save_path)

if __name__=="__main__":
    model_create_and_save()