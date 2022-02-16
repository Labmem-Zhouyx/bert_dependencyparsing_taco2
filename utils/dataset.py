import os
import random
import numpy as np
import torch
from string import punctuation
import re

import dgl
from text import text_to_sequence


class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) loads mel-spectrograms from mel files
    """
    def __init__(self, fname, hparams):
        self.symbols_lang = hparams["lang"]

        self.data_path = hparams["dataset"]["data_path"]
        self.text_cleaners = hparams["text_cleaners"]

        self.mel_dim = hparams["model"]["mel_dim"]
        self.f_list = self.files_to_list(fname)

        random.seed(hparams["train"]["seed"])
        random.shuffle(self.f_list)

    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().strip('\ufeff').split('|') #remove BOM

                # wordemb_file_path
                wordemb_path = os.path.join(self.data_path, "wordemb", "{}-wordemb-{}.npy".format(parts[1], parts[0]))
                p2widx_path = os.path.join(self.data_path, "p2widx", "{}-p2widx-{}.npy".format(parts[1], parts[0]))
                depgraph_path = os.path.join(self.data_path, "depgraph", "{}-depgraph-{}.npy".format(parts[1], parts[0]))
                # mel_file_path
                mel_path  = os.path.join(self.data_path, "mel", "{}-mel-{}.npy".format(parts[1], parts[0]))
                # text
                phone_text = parts[2]

                f_list.append([phone_text, wordemb_path, p2widx_path, depgraph_path, mel_path])

        return f_list

    def get_mel_text_pair(self, phone_text, wordemb_path, p2widx_path, depgraph_path, mel_path):
        text, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id = self.get_text(phone_text, wordemb_path, p2widx_path, depgraph_path)
        mel = self.get_mel(mel_path)
        return (text, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id, mel)

    def get_mel(self, file_path):
        #stored melspec: np.ndarray [shape=(T_out, num_mels)]
        melspec = torch.from_numpy(np.load(file_path))
        assert melspec.size(1) == self.mel_dim, (
            'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(0), self.mel_dim))

        return melspec

    def get_text(self, phone_text, wordemb_path, p2widx_path, depgraph_path):

        text_norm = torch.IntTensor(text_to_sequence(phone_text, self.text_cleaners))
        word_emb = torch.from_numpy(np.load(wordemb_path))
        phone2word_idx = torch.from_numpy(np.load(p2widx_path))
        depgraph = np.load(depgraph_path)
        nodes_list1 = torch.from_numpy(depgraph[0])
        nodes_list2 = torch.from_numpy(depgraph[1])
        deprels_id = torch.from_numpy(depgraph[2])

        return text_norm, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id

    def __getitem__(self, index):
        return self.get_mel_text_pair(*self.f_list[index])

    def __len__(self):
        return len(self.f_list)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, r):
        self.r = r

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        max_words_len = max(len(x[1]) for x in batch)
        word_padded = torch.FloatTensor(len(batch), max_words_len, batch[0][1].size(1))
        word_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            word_emb = batch[ids_sorted_decreasing[i]][1]
            word_padded[i, :word_emb.size(0)] = word_emb

        phone2word_padded = torch.LongTensor(len(batch), max_input_len)
        phone2word_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            phone2word_idx = batch[ids_sorted_decreasing[i]][2]
            phone2word_padded[i, :phone2word_idx.size(0)] = phone2word_idx

        # Construct dependency graph
        g_list = []
        for i in range(len(ids_sorted_decreasing)):
            g = dgl.graph((batch[ids_sorted_decreasing[i]][3], batch[ids_sorted_decreasing[i]][4]),
                          num_nodes=max_words_len)
            g.edata['type'] = batch[ids_sorted_decreasing[i]][5]
            g_list.append(g)
        g_batch = dgl.batch(g_list)

        # Right zero-pad mel-spec
        num_mels = batch[0][6].size(1)
        max_target_len = max([x[6].size(0) for x in batch])
        if max_target_len % self.r != 0:
            max_target_len += self.r - max_target_len % self.r
            assert max_target_len % self.r == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), max_target_len, num_mels)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][6]
            mel_padded[i, :mel.size(0), :] = mel
            gate_padded[i, mel.size(0)-1:] = 1
            output_lengths[i] = mel.size(0)

        return text_padded, input_lengths, word_padded, phone2word_padded, g_batch, mel_padded, gate_padded, output_lengths


class TextDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
    """
    def __init__(self, fname, hparams):
        self.symbols_lang = hparams["lang"]

        self.data_path = hparams["dataset"]["data_path"]
        self.text_cleaners = hparams["text_cleaners"]

        self.f_list = self.files_to_list(fname)

        random.seed(hparams["train"]["seed"])
        random.shuffle(self.f_list)

    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().strip('\ufeff').split('|') #remove BOM

                # wordemb_file_path
                wordemb_path = os.path.join(self.data_path, "wordemb", "{}-wordemb-{}.npy".format(parts[1], parts[0]))
                p2widx_path = os.path.join(self.data_path, "p2widx", "{}-p2widx-{}.npy".format(parts[1], parts[0]))
                depgraph_path = os.path.join(self.data_path, "depgraph", "{}-depgraph-{}.npy".format(parts[1], parts[0]))
                # text
                phone_text = parts[2]
                basename = parts[0]

                f_list.append([basename, phone_text, wordemb_path, p2widx_path, depgraph_path])

        return f_list

    def get_text(self, basename, phone_text, wordemb_path, p2widx_path, depgraph_path):

        text_norm = torch.IntTensor(text_to_sequence(phone_text, self.text_cleaners))
        word_emb = torch.from_numpy(np.load(wordemb_path))
        phone2word_idx = torch.from_numpy(np.load(p2widx_path))
        depgraph = np.load(depgraph_path)
        nodes_list1 = torch.from_numpy(depgraph[0])
        nodes_list2 = torch.from_numpy(depgraph[1])
        deprels_id = torch.from_numpy(depgraph[2])

        return text_norm, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id, basename

    def __getitem__(self, index):
        return self.get_text(*self.f_list[index])

    def __len__(self):
        return len(self.f_list)


class TextCollate():
    """ Zero-pads model inputs
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id, basename]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        max_input_len = int(max(input_lengths))

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(batch)):
            text = batch[i][0]
            text_padded[i, :text.size(0)] = text

        max_words_len = max(len(x[1]) for x in batch)
        word_padded = torch.FloatTensor(len(batch), max_words_len, batch[0][1].size(1))
        word_padded.zero_()
        for i in range(len(batch)):
            word_emb = batch[i][1]
            word_padded[i, :word_emb.size(0)] = word_emb

        phone2word_padded = torch.LongTensor(len(batch), max_input_len)
        phone2word_padded.zero_()
        for i in range(len(batch)):
            phone2word_idx = batch[i][2]
            phone2word_padded[i, :phone2word_idx.size(0)] = phone2word_idx

        # Construct dependency graph
        g_list = []
        for i in range(len(batch)):
            g = dgl.graph((batch[i][3], batch[i][4]),
                          num_nodes=max_words_len)
            g.edata['type'] = batch[i][5]
            g_list.append(g)
        g_batch = dgl.batch(g_list)

        basenames = []
        for i in range(len(batch)):
            basenames.append(batch[i][6])

        return (text_padded, input_lengths, word_padded, phone2word_padded, g_batch), basenames

