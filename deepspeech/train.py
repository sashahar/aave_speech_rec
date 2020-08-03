import argparse
import time
import torch
import torch.nn as nn
import json
import os
import numpy as np

from model import DeepSpeech
from dataloader import AudioDataLoader, AudioDataset, BucketingSampler
from decoder import GreedyDecoder, BeamCTCDecoder
from test import evaluate

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR', help='path to train manifest csv',
                    default="cv_valid-train-processed-supersubset_manifest_vm.csv")
parser.add_argument('--val-manifest', metavar='DIR', help='path to validation manifest csv',
                    default='cv_valid-dev-processed-supersubset_manifest_vm.csv')
parser.add_argument('--char-vocab-path', default="character_vocab.json",
                    help='Contains all characters for transcription')
parser.add_argument('--batch-size', default=10, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', default=70, type=int,
                    help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--learning-anneal', default=1.0, type=float,
                    help='Annealing applied to learning rate every epoch.  Default is no annealing')
parser.add_argument(
    '--id', type=str, help='Unique identifier for current training operation')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--pretrained', action='store_true',
                    help='Continue from pretrained weights')
parser.add_argument('--save-txt-file', default='logs/training_word_preds_subset_2_conv.csv',
                    help='Filepath to save best seen transcriptions during training')
parser.add_argument('--checkpoint', dest='checkpoint',
                    action='store_true', help='Checkpoint each epoch')
parser.add_argument('--checkpoint-best', dest='checkpointbest',
                    action='store_true', help='Checkpoint best model')
parser.add_argument('--beam-decode', action='store_true',
                    help='Type of decoder to use in model evaluation: Options are greedy decoding and beam search decoding.')
parser.add_argument('--hidden-dim', type = int, default = 512,
                    help='Size of hidden units used in deepspeech model')
parser.add_argument('--use-mfcc-features', action='store_true',
                    help='Type of decoder to use in model evaluation: Options are greedy decoding and beam search decoding.')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--ngpu', type = int, default = 1,
                    help='Number of GPUs to use during training.  a number larger than 1 parallelizes training.')

MODEL_SAVE_DIR = 'models'
LOG_DIR = 'logs/'
LOG_FILE = 'log'
SAVE_TXT_FILE = 'word_preds'
SAVE_MODEL_PARAMS = 'model_checkpoint'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_saved_model(args):
    print("Loading checkpoint model %s" % args.continue_from)
    package = torch.load(args.continue_from,
                         map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)

    # if not args.finetune:  # Don't want to restart training
    optim_state = package['optim_dict']
    start_epoch = int(package.get('epoch', 1)) - \
        1  # Index start at 0 for training
    start_iter = package.get('iteration', None)
    if start_iter is None:
        # We saved model after epoch finished, start at the next epoch.
        start_epoch += 1
        start_iter = 0
    else:
        start_iter += 1
    avg_loss = package.get('avg_loss', 0)
    hidden_dim = package.get('hidden_size', None)
    return model, optim_state, start_epoch, start_iter, avg_loss, hidden_dim


if __name__ == '__main__':

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.cuda else "cpu")
    print('using device: {}'.format(device))

    LOG_DIR += args.id

    start_epoch, start_iter, optim_state = 0, 0, None
    if args.continue_from:  # Starting from previous model
        model, optim_state, start_epoch, _, avg_loss, hidden_dim = load_saved_model(args)
        args.hidden_dim = hidden_dim
        print("previous avg loss is : ", avg_loss)
    else:
        model = DeepSpeech(args.hidden_dim, use_mfcc_features = args.use_mfcc_features)
    # if args.pretrained:  # Starting from previous model
    #     optim_state = None
    #     start_epoch = 0
    print("Hidden Size: {}, Number of params: {}".format(args.hidden_dim, DeepSpeech.get_param_size(model)))
    model = model.to(device)

    #Initialize multi GPU training if applicable
    if args.ngpu > 1:
        print("Using DataParallel to distribute across {} GPUs".format(args.ngpu))
        model = nn.DataParallel(model, list(range(args.ngpu))) #TODOL what is this second argument?

    with open(args.char_vocab_path) as label_file:
        characters = str(''.join(json.load(label_file)))

    if not args.beam_decode:
        decoder = GreedyDecoder(characters)
    else:
        decoder = BeamCTCDecoder(characters)

    beam_decoder = BeamCTCDecoder(characters)

    train_dataset = AudioDataset(
        manifest_filepath=args.train_manifest, char_vocab_path=args.char_vocab_path, use_mfcc_features= args.use_mfcc_features)
    train_eval_dataset = AudioDataset(
        manifest_filepath=args.train_manifest, char_vocab_path=args.char_vocab_path, use_mfcc_features= args.use_mfcc_features)
    val_dataset = AudioDataset(
        manifest_filepath=args.val_manifest, char_vocab_path=args.char_vocab_path, use_mfcc_features= args.use_mfcc_features)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)

    train_loader = AudioDataLoader(train_dataset, batch_sampler=train_sampler, num_workers = 10)
    train_eval_loader = AudioDataLoader(
        train_eval_dataset, batch_size=args.batch_size, num_workers = 10)
    val_loader = AudioDataLoader(val_dataset, batch_size=args.batch_size, num_workers = 10)

    lr = args.lr
    print("using learning rate: ", lr)
    # nesterov=True, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # load previous optimizer state if using a pretrained model
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    #loss_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs)
    loss_results, wer_dev_results, cer_dev_results, wer_train_results, cer_train_results = [], [], [], [], []
    batch_time = AverageMeter()
    losses = AverageMeter()
    ctc_loss = nn.CTCLoss()

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    loss_log_file = LOG_DIR + "/" + LOG_FILE + "_loss.csv"
    temp_high_loss_file = LOG_DIR + "/" +  "high_loss_examples.csv"
    cer_wer_log_file = LOG_DIR + "/" + LOG_FILE + "_er.csv"
    beam_decode_log_file = LOG_DIR + "/" + LOG_FILE + "_beam_er.csv"
    save_word_preds_file_train = LOG_DIR + "/" + SAVE_TXT_FILE + "_train.csv"
    save_word_preds_file_val = LOG_DIR + "/" + SAVE_TXT_FILE + "_val.csv"
    save_word_preds_file_train_best = LOG_DIR + "/" + SAVE_TXT_FILE + "_train_best.csv"
    save_word_preds_file_val_best = LOG_DIR + "/" + SAVE_TXT_FILE + "_val_best.csv"
    save_model_params_file_val_best = LOG_DIR + "/" + SAVE_MODEL_PARAMS + "_val_best.pth"
    save_model_params_file_train_best = LOG_DIR + "/" + SAVE_MODEL_PARAMS + "_train_best.pth"

    if not args.continue_from:
        np.savetxt(loss_log_file, np.array(
            ['epoch,loss']), fmt="%s", delimiter=",")
        np.savetxt(temp_high_loss_file, np.array(
            ['epoch,iter,loss,example']), fmt="%s", delimiter=",")
        np.savetxt(cer_wer_log_file, np.array(
            ['epoch,loss,train_cer,train_wer, dev_cer,dev_wer']), fmt="%s", delimiter=",")
        np.savetxt(beam_decode_log_file, np.array(
            ['epoch,loss,train_cer,train_wer,dev_cer,dev_wer,dev_beam_cer,dev_beam_wer']), fmt="%s", delimiter=",")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        avg_loss = 0

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break

            #CLEAR CACHE
            torch.cuda.empty_cache()
            # try:
            inputs, targets, input_sizes, target_sizes, filenames = data
            total_length = max(input_sizes)
            print("total length outside is: ", total_length)

            inputs = inputs.to(device)
            targets = targets.to(device)

            out, output_sizes = model(inputs, input_sizes, total_length)
            print("Outside: input size", inputs.size(),
            "output_size", out.size())

            out = out.transpose(0, 1)  # TxNxH

            #Work around on CTCLoss bug
            #https://github.com/pytorch/pytorch/issues/22234
            with torch.backends.cudnn.flags(enabled=False):
                loss = ctc_loss(out, targets, output_sizes,
                                target_sizes).to(device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 350)
            optimizer.step()

            avg_loss += float(loss.detach())
            losses.update(float(loss.detach()), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            #TEMPORARY
            loss_num = float(loss.cpu().detach())
            if loss_num > 50:
                with open(temp_high_loss_file, "a") as file:
                    file.write("{},{},{},{}\n".format(epoch, i, loss_num, filenames[0]))

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, loss=losses))
            torch.empty

        avg_loss /= len(train_sampler)
        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        loss_results.append(avg_loss)

        with open(loss_log_file, "a") as file:
            file.write("{},{}\n".format(epoch, avg_loss))

        if (epoch % 3 == 0) or (epoch % 5 == 0):
            with torch.no_grad():
                train_wer, train_cer, out_train_data, out_train_text = evaluate(test_loader=train_eval_loader, device=device, model=model, decoder=decoder, target_decoder=decoder)
                wer, cer, output_data, output_text = evaluate(
                    test_loader=val_loader, device=device, model=model, decoder=decoder, target_decoder=decoder)
                beam_wer, beam_cer, beam_output_data, beam_output_text = evaluate(
                    test_loader=val_loader, device=device, model=model, decoder=beam_decoder, target_decoder=decoder)
                # To evaluate on test set: evaluate(test_loader=train_eval_loader, device=device,model=model,decoder=decoder,target_decoder=decoder) #Edited line to evaluate on train set --> evaluate(test_loader=val_loader, device=device,model=model,decoder=decoder,target_decoder=decoder)
            wer_train_results.append(train_wer)
            cer_train_results.append(train_cer)
            wer_dev_results.append(wer)
            #cer_results[epoch] = cer
            cer_dev_results.append(cer)
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'
                  'Average Train WER {t_wer:.3f}\t'
                  'Average Train CER {t_cer:.3f}\t'.format(
                      epoch + 1, wer=wer, cer=cer, t_wer=train_wer, t_cer=train_cer))
            with open(cer_wer_log_file, "a") as file:
                file.write("{},{},{},{},{},{}\n".format(
                    epoch, avg_loss, train_cer, train_wer, cer, wer))
            with open(beam_decode_log_file, "a") as file:
                file.write("{},{},{},{},{},{},{},{}\n".format(
                    epoch, avg_loss, train_cer, train_wer, cer, wer, beam_cer, beam_wer))

            np.savetxt(save_word_preds_file_train, out_train_text, fmt="%s", delimiter=",")
            np.savetxt(save_word_preds_file_val, output_text, fmt="%s", delimiter=",")

            if cer <= min(cer_dev_results):  # if True:
                print("New best achieved, writing output to: {}".format(
                    save_word_preds_file_val_best))
                # np.savetxt(save_word_preds_file_train_best, out_train_text, fmt="%s", delimiter=",")
                np.savetxt(save_word_preds_file_val_best, output_text, fmt="%s", delimiter=",")
                if args.checkpoint:
                    print("Saving checkpoint model to %s" % save_model_params_file_val_best)
                    torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                    loss_results=loss_results,
                                                    wer_results=wer_dev_results, cer_results=cer_dev_results, avg_loss=avg_loss),
                                                    save_model_params_file_val_best)
            if train_cer <= min(cer_train_results):
                print("New best achieved, writing output to: {}".format(
                    save_word_preds_file_train_best))
                np.savetxt(save_word_preds_file_train_best, out_train_text, fmt="%s", delimiter=",")
                if args.checkpoint:
                    print("Saving checkpoint model to %s" % save_model_params_file_train_best)
                    torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                    loss_results=loss_results,
                                                    wer_results=wer_dev_results, cer_results=cer_dev_results, avg_loss=avg_loss),
                                                    save_model_params_file_train_best)

        # anneal lr
        lr = lr / args.learning_anneal
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=lr))

        print("Shuffling batches...")
        train_sampler.shuffle(epoch)
