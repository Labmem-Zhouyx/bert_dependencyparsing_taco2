import os
import random
import numpy as np
import torch
from string import punctuation
import re

import dgl

from text import text_to_sequence
from text.dependency_relations import deprel_labels_to_id


_puncts_dict = {"，":",", "。":".", "？":"?", "！":"!"}
class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) loads mel-spectrograms from mel files
    """
    def __init__(self, fname, hparams, nlp, bert, tokenizer):
        self.symbols_lang = hparams["lang"]

        self.melfile_path = hparams["dataset"]["melfile_path"]
        self.lexicon = self.read_lexicon(hparams["dataset"]["lexicon_path"])
        self.text_cleaners = hparams["text_cleaners"]

        self.mel_dim = hparams["model"]["mel_dim"]
        self.f_list = self.files_to_list(fname)

        self.tokenizer = tokenizer
        self.nlp = nlp
        self.bert = bert

        random.seed(hparams["train"]["seed"])
        random.shuffle(self.f_list)

    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().strip('\ufeff').split('|') #remove BOM
                # mel_file_path
                path  = os.path.join(self.melfile_path, "{}-mel-{}.npy".format(parts[1], parts[0]))
                # text
                phone_text = parts[2]
                raw_text = parts[3]
                f_list.append([phone_text, raw_text, path])

        return f_list

    def get_mel_text_pair(self, phone_text, raw_text, file_path):
        text, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id = self.get_text(phone_text, raw_text)
        mel = self.get_mel(file_path)
        return (text, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id, mel)

    def get_mel(self, file_path):
        #stored melspec: np.ndarray [shape=(T_out, num_mels)]
        melspec = torch.from_numpy(np.load(file_path))
        assert melspec.size(1) == self.mel_dim, (
            'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(0), self.mel_dim))

        return melspec

    def get_text(self, phone_text, raw_text):
        subwords = self.tokenizer.tokenize(raw_text)
        subword_idx = torch.tensor([self.tokenizer.convert_tokens_to_ids(subwords)])

        with torch.no_grad():
            bert_emb = self.bert(subword_idx)[0][11][0]

        words, nodes_list1, nodes_list2, deprels_id = self.get_dependencyparsing(raw_text)
        # print(words, nodes_list1, nodes_list2, deprels_id)
        word_emb = self.get_word_embedding(subwords, words, bert_emb)

        if self.symbols_lang == 'zh':
            phone, phone2word_idx = self.preprocess_mandarin(phone_text, words)
        else:
            phone, phone2word_idx = self.preprocess_english(phone_text, words)

        text_norm = torch.IntTensor(text_to_sequence(phone, self.text_cleaners))
        phone2word_idx = torch.IntTensor(phone2word_idx)
        nodes_list1 = torch.IntTensor(nodes_list1)
        nodes_list2 = torch.IntTensor(nodes_list2)
        deprels_id = torch.IntTensor(deprels_id)

        return text_norm, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id

    def read_lexicon(self, lex_path):
        lexicon = {}
        with open(lex_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def preprocess_english(self, text, refer_words):

        phones = []
        phone2word_idx = []
        words = re.split(r"([,:;.()\-\?\!\s+])", text)
        index = 0
        tmp = ''
        for w in words:
            if w == " " or w == "":
                continue
            if w.lower() in self.lexicon:
                phones += self.lexicon[w.lower()]
                phone2word_idx.extend([index] * len(self.lexicon[w.lower()]))
            elif w in list(",:;.()?!-"):
                phones.append(w)
                phone2word_idx.extend([index])
            tmp += w
            if refer_words[index] == tmp:
                index += 1
                tmp = ''

        assert index == len(refer_words)
        assert len(phones) == len(phone2word_idx)
        phones = "{" + " ".join(phones) + "}"
        # print("Raw Text Sequence: {}".format(text))
        # print("Phoneme Sequence: {}".format(phones))
        # print("Phone_word_id: ", phone2word_idx)
        return phones, phone2word_idx

    def preprocess_mandarin(self, text, refer_words):

        phones = []
        phone2word_idx = []
        pinyins = re.split(r"([,./\-\?\!\s+])", text)
        index = 0
        tmpcnt = 0
        for p in pinyins:
            if p in self.lexicon:
                phones += self.lexicon[p]
                phone2word_idx.extend([index] * len(self.lexicon[p]))
                tmpcnt += 1
                if tmpcnt == len(refer_words[index]):
                    index += 1
                    tmpcnt = 0
            elif index < len(refer_words) and refer_words[index] in list("，。？！"):
                    phones.append(_puncts_dict[refer_words[index]])
                    phone2word_idx.extend([index])
                    index += 1
            else:
                continue

        assert index == len(refer_words)
        assert len(phones) == len(phone2word_idx)
        phones = "{" + " ".join(phones) + "}"
        # print("Raw Text Sequence: {}".format(text))
        # print("Phoneme Sequence: {}".format(phones))
        # print("Phone_word_id: ", phone2word_idx)
        return phones, phone2word_idx

    def get_dependencyparsing(self, sen):
        doc = self.nlp(sen)
        words = []
        heads = []
        deprels = []
        upos = []
        for sent in doc.sentences:
            for word in sent.words:
                words.append(word.text)
                heads.append(word.head)
                deprels.append(word.deprel)
                upos.append(word.upos)

        List1 = [i for i in range(len(heads)) if heads[i] != 0]
        List2 = [heads[i] - 1 for i in range(len(heads)) if heads[i] != 0]
        deprels_id = [deprel_labels_to_id[deprels[i]] for i in range(len(heads)) if heads[i] != 0]

        # print("CHECK text:", sen)
        # print("CHECK List1:", List1)
        # print("CHECK List2:", List2)
        # print("CHECK deprel:", deprel)
        return words, List1, List2, deprels_id

    def get_word_embedding(self, subwords, words, bert_emb):
        tmp = ""
        index = 0
        last_i = 0
        word_emb = torch.zeros(len(words), bert_emb.shape[1])
        for i in range(len(subwords)):
            tmp += subwords[i].replace("#", "").replace("[UNK]", "_")
            if len(tmp) == len(words[index]):
                word_emb[index] = torch.mean(bert_emb[last_i:i+1], axis=0)
                index += 1
                tmp = ""
                last_i = i + 1

        assert index == len(words)
        return word_emb

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

