import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from dataloader import audio_conf

RNN_HIDDEN_SIZE = 512

class CustomDataParallel(nn.Module):
    def __init__(self, model, device_ids = None):
        super(CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model, device_ids).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return F.log_softmax(input_, dim=-1)

class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        # self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, total_length):
        #TxNxH (seq_len, batch, feature_dim)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, batch_first=True)
        self.rnn.flatten_parameters()
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length = total_length)
        if self.bidirectional:
            # (TxNxH*2) -> (TxNxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

class DeepSpeech(nn.Module):
    def __init__(self, rnn_hidden_size, use_mfcc_features = False, sample_rate=audio_conf['sample_rate'], nb_layers=4, window_size=audio_conf['window_size']):
        super().__init__()

        self.sample_rate = sample_rate
        self.rnn_hidden_size = rnn_hidden_size
        self.window_size = window_size
        self.use_mfcc_features = use_mfcc_features

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
        ))

        self.mfcc_fc = nn.Sequential(
            nn.Linear(40 * 19, self.rnn_hidden_size),
            nn.BatchNorm1d(self.rnn_hidden_size),
            nn.Hardtanh(0, 20, inplace=True)
        )

        if self.use_mfcc_features:
            rnn_input_size = rnn_hidden_size
        else:
            rnn_input_size = int(math.floor((audio_conf['n_fft']) / 2) + 1)
        #Calculate output size of convolutional layers
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []

        #self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=self.rnn_hidden_size, bidirectional=True, bias=True)
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=nn.LSTM,
                       bidirectional=True, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=nn.LSTM,
                           bidirectional=True)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, 29, bias=False)
        )

        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths, total_length):
        #X has shape: batch x 1 (num_channels) x n_fft (constant over all batches) x padded_seq_len
        output_lengths = self.get_seq_lens(lengths)

        print("Input shape: ", x.shape)

        if self.use_mfcc_features:
            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) #Collapse feature dimension NxFxT
            x = x.transpose(1, 2).contiguous() #NxTxF
            n, t = x.size(0), x.size(1) #NxTxF --> (NxT)xF
            x = x.view(n * t, -1)

            x = self.mfcc_fc(x)

            x = x.view(n, t, -1) #(NxT)xH --> NxTxH
            print("After fully connected shape: ", x.shape)
        else:
            x, _ = self.conv(x, output_lengths) #X has shape: batch x 32 (num_channels) x rnn_input_size//32 x f(padded_seq_len)
            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            x = x.transpose(1, 2).contiguous()  # NxTxH (batch dim first)

        for rnn in self.rnns:
            x = rnn(x, output_lengths, total_length)
        #Output of RNN is batch first #NxTxH2

        x = x.transpose(0, 1).contiguous() # TxNxH2, where H2 is self.rnn_hidden_size
        # T*N*H -> (T*N)*H
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)

        x = self.fc(x)

        # (T*N)*H -> T*N*H
        x = x.view(t, n, -1)
        x = x.transpose(0, 1)

        x = self.inference_softmax(x)

        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None):
        package = {
            'hidden_size': model.rnn_hidden_size,
            'sample_rate': model.sample_rate,
            'window_size': model.window_size,
            'use_mfcc_features': model.use_mfcc_features,
            'state_dict': model.state_dict(),
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        return package

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @classmethod
    def load_model_package(cls, package):
        mfcc = package.get('use_mfcc_features', False)
        model = cls(rnn_hidden_size=package['hidden_size'],
                    use_mfcc_features = mfcc,
                        sample_rate= package['sample_rate'],
                        window_size=package['window_size'])
        model.load_state_dict(package['state_dict'])
        return model

class SimpleNN(nn.Module):
    def __init__(self, input_sz, hidden_size = 256):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(input_sz, hidden_size)
        #features Needs to be 1000 for resnet #Needs to be 4096 for vgg #Needs to be 512 when doing CNN
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)

        return out

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, hidden_size = 128, bidirectional = False, vocab_size = 29, output_size=2):
        super(SimpleLSTMClassifier,self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, input, input_lengths):
        '''
        input: ionput tensor for the LSTM classifier.  Expect it to have dimensions
        TxNxH, where
        T = sequence length
        N = batch size
        H = vocab size (can be thought of as the embedding in this application)
        '''
        #LSTM expects input of size: (seq_len, batch, input_size)
        T,N,H = input.shape
        #input = input.permute(1, 0, 2)
        #print(input.shape)

        x = nn.utils.rnn.pack_padded_sequence(input, input_lengths) #Transforms from #NxTxH to (sum(T_i)xH)

        #todo: need to apply packed padded sequence here?
        #if we only supply x, and not c), h0, then ht and ct will be none!

        #print("after pack padded: ", x.shape)

        packed_out, (ht, ct) = self.lstm(x)
        # ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)

        #ignore dropout for now
        # output = self.dropout_layer(ht[-1])
        out = self.hidden2out(ht) #packed_out.data)
        out = out.view(N, -1) #reshape such that shape[0] = batch size
        #no softmax do to specifications of CrossEntropyLoss
        return out

    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None):
        package = {
            'hidden_size': model.hidden_size,
            # 'sample_rate': model.sample_rate,
            # 'window_size': model.window_size,
            'state_dict': model.state_dict(),
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        return package

    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    @classmethod
    def load_model_package(cls, package):
        model = cls(hidden_size=package['hidden_size'])
        model.load_state_dict(package['state_dict'])
        return model
