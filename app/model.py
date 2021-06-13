import time
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.data.utils import get_tokenizer
from typing import Tuple
from torch import Tensor
from data import get_filepaths, build_vocab


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))
        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs




def build_model(src_vocab_size, tgt_vocab_size, device):
    MODEL_PATH = "/opt/ml/model"
    
    INPUT_DIM = src_vocab_size
    OUTPUT_DIM = tgt_vocab_size
    ENC_EMB_DIM = 12
    DEC_EMB_DIM = 12
    ENC_HID_DIM = 24
    DEC_HID_DIM = 24
    ATTN_DIM = 2
    # 1,202,392
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model


def translate(sentence: str) -> str:
    """Takes a sentence in French and returns it's translation in English.
    """
    start_time = time.time()
    MAX_LEN = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fr_tokenizer = get_tokenizer('spacy', language='fr')
    en_tokenizer = get_tokenizer('spacy', language='en')

    try:
        with open("/opt/ml/en_vocab.pkl", "rb") as f:
            en_vocab = pickle.load(f)
        with open("/opt/ml/fr_vocab.pkl", "wb") as f:
            fr_vocab = pickle.load(f)
    except:
        train_filepahts, _, _ = get_filepaths()
        fr_vocab = build_vocab(train_filepahts[0], fr_tokenizer)
        en_vocab = build_vocab(train_filepahts[1], en_tokenizer)

    tokens = fr_tokenizer(sentence)
    x = torch.tensor([fr_vocab.stoi[word] for word in tokens])
    src = x.view(-1, 1).to(device)
    tgt = torch.tensor([en_vocab['<sos>']] + [0] * MAX_LEN).view(-1, 1).to(device)
    
    # Load the model    
    model = build_model(len(fr_vocab), len(en_vocab), device)

    output = model(src, tgt, 0)

    print(f"output shape: {output.shape} | vocab size: {len(en_vocab)}")
    
    translation = []
    for i in range(output.size(0)):
        idx = torch.argmax(output[i][0])
        word = en_vocab.itos[idx]
        if word == "<unk>": continue
        if word == "<eos>": break
        translation.append(word)

    end_time = time.time()
    print(f"⌛ Time to process: {end_time - start_time}")
    return " ".join(translation)