import torch
import tqdm
from src.exception.exception import ExceptionNetwork, sys


def model_validation(valid_dataloader, loss_fn, Model, device):
    try:
        Model.eval()

        progress_bar = tqdm.tqdm(range(len(valid_dataloader)), "Validation Process")

        with torch.no_grad():

            total_values_valid = 0
            correct_values_valid = 0
            loss_values_valid = 0

            for batch_valid, (input_data_valid, output_data_valid) in enumerate(valid_dataloader):

                input_encoder_valid = input_data_valid.to(device)
                output_decoder_valid = output_data_valid.to(device)
                
                # initial input transformer decoder
                initial_input_decoder=torch.zeros_like(output_decoder_valid).to(device)
                initial_input_decoder[:,0]=output_decoder_valid[:,0] # Add start token to input decoder
                
                # To remove start token from out decoder
                output_decoder_valid = output_decoder_valid[:, 1:]

                output_valid = Model(input_encoder_valid, initial_input_decoder, output_decoder_valid)

                output_decoder_valid = output_decoder_valid[:,:output_valid.shape[1]]
                loss_valid = loss_fn(output_valid.reshape(-1, output_valid.shape[-1]), output_decoder_valid.reshape(-1))

                _, pred_valid = torch.max(output_valid, -1)
                correct_values_valid += (pred_valid ==output_decoder_valid).sum().item()
                total_values_valid += (output_decoder_valid.size(0)* output_decoder_valid.size(1))
                loss_values_valid += loss_valid.item()

                progress_bar.update(1)
                
            total_loss = loss_values_valid/(batch_valid+1)
            total_acc = (correct_values_valid/total_values_valid)*100

            progress_bar.set_postfix({"valid_acc": total_acc,
                                      "valid_loss": total_loss})

        return total_loss, total_acc

    except Exception as e:
        raise ExceptionNetwork(e, sys)
