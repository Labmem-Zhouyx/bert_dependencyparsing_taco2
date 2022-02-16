import os
import numpy as np
import torch
import stanza
import re
from tqdm import tqdm
from pytorch_pretrained_bert import BertModel, BertTokenizer
from text.dependency_relations import deprel_labels_to_id


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

_puncts_dict = {"，":",", "。":".", "？":"?", "！":"!"}
lexicon_en = read_lexicon("lexicon/librispeech-lexicon.txt")
lexicon_py = read_lexicon("lexicon/pinyin-lexicon-r.txt")

def get_text(tokenizer, bert, nlp, lang, phone_text, raw_text):
    subwords = tokenizer.tokenize(raw_text)
    subword_idx = torch.tensor([tokenizer.convert_tokens_to_ids(subwords)])

    with torch.no_grad():
        bert_emb = bert(subword_idx)[0][11][0]

    words, nodes_list1, nodes_list2, deprels_id = get_dependencyparsing(nlp, raw_text)
    # print(words, nodes_list1, nodes_list2, deprels_id)
    word_emb = get_word_embedding(subwords, words, bert_emb)

    if lang == 'zh':
        phone, phone2word_idx = preprocess_mandarin(phone_text, words)
    else:
        phone, phone2word_idx = preprocess_english(phone_text, words)

    return phone, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id

def preprocess_english(text, refer_words):

    phones = []
    phone2word_idx = []
    words = re.split(r"([,:;.()\-\?\!\s+])", text)
    index = 0
    tmp = ''
    for w in words:
        if w == " " or w == "":
             continue
        if w.lower() in lexicon_en:
            phones += lexicon_en[w.lower()]
            phone2word_idx.extend([index] * len(lexicon_en[w.lower()]))
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

def preprocess_mandarin(text, refer_words):

    phones = []
    phone2word_idx = []
    pinyins = re.split(r"([,./\-\?\!\s+])", text)
    index = 0
    tmpcnt = 0
    for p in pinyins:
        if p in lexicon_py:
            phones += lexicon_py[p]
            phone2word_idx.extend([index] * len(lexicon_py[p]))
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

def get_dependencyparsing(nlp, sen):
    doc = nlp(sen)
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

def get_word_embedding(subwords, words, bert_emb):
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

def generate_english(filepath, outpath, datapath):
    print("Processing {}...".format(filepath))
    # stanza.download("en", "path")
    nlp = stanza.Pipeline("en", "/apdcephfs/private_yatsenzhou/pretrained/stanza/stanza_en")
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
    bert = BertModel.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-uncased/bert-base-uncased-vocab.txt")

    os.makedirs((os.path.join(datapath, "wordemb")), exist_ok=True)
    os.makedirs((os.path.join(datapath, "p2widx")), exist_ok=True)
    os.makedirs((os.path.join(datapath, "depgraph")), exist_ok=True)

    out = []
    with open(filepath, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().strip('\ufeff').split('|')  # remove BOM
            index, speaker, phone_text, raw_text = parts[0], parts[1], parts[2], parts[3]

            phone, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id = get_text(tokenizer, bert, nlp, "en", phone_text, raw_text)
            out.append(index + "|" + speaker + "|" + phone + "|" + phone_text + "\n")

            wordemb_filename = "{}-wordemb-{}.npy".format(speaker, index)
            np.save(os.path.join(datapath, "wordemb", wordemb_filename), word_emb.cpu().detach().numpy())

            p2widx_filename = "{}-p2widx-{}.npy".format(speaker, index)
            np.save(os.path.join(datapath, "p2widx", p2widx_filename), np.asarray(phone2word_idx))

            depgraph_filename = "{}-depgraph-{}.npy".format(speaker, index)
            np.save(os.path.join(datapath, "depgraph", depgraph_filename), np.asarray([nodes_list1, nodes_list2, deprels_id]))

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("".join(out))


def generate_chinese(filepath, outpath, datapath):
    print("Processing {}...".format(filepath))
    # stanza.download("zh", "path")
    nlp = stanza.Pipeline("zh", "/apdcephfs/private_yatsenzhou/pretrained/stanza/stanza_zh")
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
    bert = BertModel.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-chinese/bert-base-chinese-vocab.txt")

    os.makedirs((os.path.join(datapath, "wordemb")), exist_ok=True)
    os.makedirs((os.path.join(datapath, "p2widx")), exist_ok=True)
    os.makedirs((os.path.join(datapath, "depgraph")), exist_ok=True)

    out = []
    with open(filepath, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().strip('\ufeff').split('|')  # remove BOM
            index, speaker, phone_text, raw_text = parts[0], parts[1], parts[2], parts[3]

            phone, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id = get_text(tokenizer, bert, nlp, "zh", phone_text, raw_text)
            out.append(index + "|" + speaker + "|" + phone + "|" + phone_text + "\n")

            wordemb_filename = "{}-wordemb-{}.npy".format(speaker, index)
            np.save(os.path.join(datapath, "wordemb", wordemb_filename), word_emb.cpu().detach().numpy())

            p2widx_filename = "{}-p2widx-{}.npy".format(speaker, index)
            np.save(os.path.join(datapath, "p2widx", p2widx_filename), np.asarray(phone2word_idx))

            depgraph_filename = "{}-depgraph-{}.npy".format(speaker, index)
            np.save(os.path.join(datapath, "depgraph", depgraph_filename), np.asarray([nodes_list1, nodes_list2, deprels_id]))

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("".join(out))


if __name__ == '__main__':
    # generate_chinese("preprocessed_data/DataBaker/train_grapheme.txt", "preprocessed_data/DataBaker/train.txt", "/data/training_data/preprocessed_data/DataBaker_16k/")
    # generate_chinese("preprocessed_data/DataBaker/val_grapheme.txt", "preprocessed_data/DataBaker/val.txt", "/data/training_data/preprocessed_data/DataBaker_16k/")
    # generate_chinese("preprocessed_data/DataBaker/test_grapheme.txt", "preprocessed_data/DataBaker/test.txt", "/data/training_data/preprocessed_data/DataBaker_16k/")
    generate_english("preprocessed_data/LJSpeech/train_grapheme.txt", "preprocessed_data/LJSpeech/train.txt", "/data/training_data/preprocessed_data/LJSpeech/")
    # generate_english("preprocessed_data/LJSpeech/val_grapheme.txt","preprocessed_data/LJSpeech/val.txt", "/data/training_data/preprocessed_data/LJSpeech/")
    # generate_english("preprocessed_data/LJSpeech/test_grapheme.txt", "preprocessed_data/LJSpeech/test.txt", "/data/training_data/preprocessed_data/LJSpeech/")



