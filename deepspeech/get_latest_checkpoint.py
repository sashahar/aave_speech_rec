# from train import load_saved_model
# from model import DeepSpeech
import argparse

parser = argparse.ArgumentParser(description='Model Checkpoint Conversion')
parser.add_argument('--checkpoint-src', default='',
                    help='Absolute to checkpoint that you would like to convert to latest checkpoint format')

if __name__ == '__main__':
    print("HERE")

    args = parser.parse_args()
    if args.checkpoint_src == '' or not os.path.exists(args.checkpoint_src):
        sys.exit('You must specify a valid checkpoint src path')


    print("Loading checkpoint from: {}".format(args.checkpoint_src))
    model, optim_state, epoch, avg_loss, hidden_dim = load_saved_model(save_model_params_file_latest)
    print("Previous checkpoint saved at epoch: ", epoch - 1)

    epoch = epoch - 2
    print("new epoch is: ", epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # load previous optimizer state if using a pretrained model
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    destination_path = '/'.join(args.checkpoint_src.split("/")[:-1] + ['model_checkpoint_latest.pth'])
    print(destination_path)
    # torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch),
    #                                 save_model_params_file_train_best)
