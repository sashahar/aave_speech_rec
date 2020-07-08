import argparse
import time
import torch
import torch.nn as nn
import json
import os
import numpy as np

from model import DeepSpeech
from dataloader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
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

MODEL_SAVE_DIR = 'models'
LOG_DIR = 'logs/'
LOG_FILE = 'log'
SAVE_TXT_FILE = 'word_preds'
SAVE_MODEL_PARAMS = 'model_checkpoint'
LOG_FILE_MEMORY_USAGE = 'memory_usage'


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
    avg_loss = int(package.get('avg_loss', 0))
    return model, optim_state, start_epoch, start_iter, avg_loss


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print('using device: {}'.format(device))

    LOG_DIR += args.id
    print("using pretrained model: ", args.pretrained)

    start_epoch, start_iter, optim_state = 0, 0, None
    if args.continue_from:  # Starting from previous model
        model, optim_state, start_epoch, _,_ = load_saved_model(args)
    if args.pretrained:  # Starting from previous model
        optim_state = None
        start_epoch = 0
    else:
        model = DeepSpeech()
    print("number of params: ", DeepSpeech.get_param_size(model))
    model = model.to(device)

    with open(args.char_vocab_path) as label_file:
        characters = str(''.join(json.load(label_file)))
    decoder = GreedyDecoder(characters)

    train_dataset = SpectrogramDataset(
        manifest_filepath=args.train_manifest, char_vocab_path=args.char_vocab_path)
    train_eval_dataset = SpectrogramDataset(
        manifest_filepath=args.train_manifest, char_vocab_path=args.char_vocab_path)
    val_dataset = SpectrogramDataset(
        manifest_filepath=args.val_manifest, char_vocab_path=args.char_vocab_path)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)

    train_loader = AudioDataLoader(train_dataset, batch_sampler=train_sampler)
    train_eval_loader = AudioDataLoader(
        train_eval_dataset, batch_size=args.batch_size)
    val_loader = AudioDataLoader(val_dataset, batch_size=args.batch_size)

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
    cer_wer_log_file = LOG_DIR + "/" + LOG_FILE + "_er.csv"
    #TEMPORARY
    temp_val_cer_wer_log_file = LOG_DIR + "/" + LOG_FILE + "_temp_val_er.csv"
    save_word_preds_file_train = LOG_DIR + "/" + SAVE_TXT_FILE + "_train.csv"
    save_word_preds_file_val = LOG_DIR + "/" + SAVE_TXT_FILE + "_val.csv"
    save_word_preds_file_train_best = LOG_DIR + "/" + SAVE_TXT_FILE + "_train_best.csv"
    save_word_preds_file_val_best = LOG_DIR + "/" + SAVE_TXT_FILE + "_val_best.csv"
    save_model_params_file_val_best = LOG_DIR + "/" + SAVE_MODEL_PARAMS + "_val_best.pth"
    save_model_params_file_train_best = LOG_DIR + "/" + SAVE_MODEL_PARAMS + "_train_best.pth"
    save_model_memory_usage = LOG_DIR + "/" + LOG_FILE_MEMORY_USAGE + ".csv"
    np.savetxt(loss_log_file, np.array(
        ['epoch,loss']), fmt="%s", delimiter=",")
    np.savetxt(cer_wer_log_file, np.array(
        ['epoch,loss,train_cer, train_wer, dev_cer,dev_wer']), fmt="%s", delimiter=",")
    #TEMPORARY
    np.savetxt(temp_val_cer_wer_log_file, np.array(
        ['epoch,loss,dev_cer,dev_wer']), fmt="%s", delimiter=",")
    np.savetxt(save_model_memory_usage, np.array(
        ['epoch,iter, memory_allocated, max_memory_allocated, max_input_len']), fmt="%s", delimiter=",")

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
            inputs, targets, input_sizes, target_sizes, _ = data

            inputs = inputs.to(device)
            targets = targets.to(device)

            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
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

        #TEMPORARY: Log validation CER,WER every epoch
        if (epoch % 2 == 0) and (epoch % 10 != 0):
            wer, cer, output_data, output_text = evaluate(
                test_loader=val_loader, device=device, model=model, decoder=decoder, target_decoder=decoder)
            with open(temp_val_cer_wer_log_file, "a") as file:
                file.write("{},{},{},{}\n".format(
                    epoch, avg_loss, cer, wer))

        if epoch % 10 == 0:
            with torch.no_grad():
                train_wer, train_cer, out_train_data, out_train_text = evaluate(test_loader=train_eval_loader, device=device, model=model, decoder=decoder, target_decoder=decoder)
                wer, cer, output_data, output_text = evaluate(
                    test_loader=val_loader, device=device, model=model, decoder=decoder, target_decoder=decoder)
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
