import torch
import tqdm
from src.exception.exception import ExceptionNetwork,sys


def model_training(train_dataloader, optimizer, loss_fn, Model,device):
    try: 
        Model.train()
        
        progress_bar=tqdm.tqdm(range(len(train_dataloader)),"Training Progress")
        
        total_values_train=0
        correct_values_train=0
        loss_values_train=0
        
        for batch_train,(input_data_train,output_data_train) in enumerate(train_dataloader):
            
            input_encoder_train=input_data_train.to(device)
            
            output_decoder_train=output_data_train.to(device)
            
            # initial input transformer decoder
            initial_input_decoder=torch.zeros_like(output_data_train).to(device)
            initial_input_decoder[:,0]=output_data_train[:,0] # Add start token to input decoder
            
            output_decoder_train=output_decoder_train[:,1:] # To remove start token from out decoder
            optimizer.zero_grad()
            output_train=Model(input_encoder_train,initial_input_decoder,output_decoder_train)
            
            # We Clip target data as output transformer because we stopped model when we reached stop token but target values is padded max len so we need to clip.
            output_decoder_train=output_decoder_train[:,:output_train.shape[1]]
            
            loss_train=loss_fn(output_train.reshape(-1,output_train.shape[-1]),output_decoder_train.reshape(-1))
            loss_train.backward()
            optimizer.step()
            
            
            _,pred_train=torch.max(output_train,-1)
            correct_values_train+=(pred_train==output_decoder_train).sum().item()
            total_values_train+=(output_decoder_train.size(0)*output_decoder_train.size(1))
            loss_values_train+=loss_train.item()
            
            progress_bar.update(1)
        
        
        total_loss = loss_values_train/(batch_train+1)
        total_acc = (correct_values_train/total_values_train)*100
        
        progress_bar.set_postfix({"train_acc":total_acc,
                                  "train_loss":total_loss})
                
        return total_loss, total_acc
   
    except Exception as e:
            raise ExceptionNetwork(e,sys)
    

        