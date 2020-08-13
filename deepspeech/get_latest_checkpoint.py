from train import load_saved_model
from model import DeepSpeech
import argparse
import os
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Model Checkpoint Conversion')
parser.add_argument('--checkpoint-src', default='',
                    help='Absolute to checkpoint that you would like to convert to latest checkpoint format')

if __name__ == '__main__':
    print("HERE")

    args = parser.parse_args()
    if args.checkpoint_src == '' or not os.path.exists(args.checkpoint_src):
        sys.exit('You must specify a valid checkpoint src path')


    print("Loading checkpoint from: {}".format(args.checkpoint_src))
    model, optim_state, epoch, avg_loss, hidden_dim = load_saved_model(args.checkpoint_src)

    epoch = epoch - 2
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001) #lr will be overwritten by state dict
    optimizer.load_state_dict(optim_state)

    destination_path = '/'.join(args.checkpoint_src.split("/")[:-1] + ['model_checkpoint_latest.pth'])
    print("Writing new checkpoint to:", destination_path)
    torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch),
            destination_path)
