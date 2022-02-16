import os
import time
import argparse
import json
import yaml

import torch
from torch.utils.data import DataLoader
import dgl
from scipy.io import wavfile
from utils.dataset import TextDataset, TextCollate
from train import create_model

import stanza
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pypinyin import pinyin, Style
from text import text_to_sequence
from vocoder import get_vocoder, vocoder_infer
from text_preprocessing import get_text


def load_checkpoint(model, checkpoint_path):
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model

def synthesize_batch(hparams, model, batch):
    with torch.no_grad():
        inputs, basenames = batch
        predicts = model.inference(inputs)
        wav_predictions = vocoder_infer(predicts[1].transpose(1, 2), vocoder, hparams)
        for wav, basename in zip(wav_predictions, basenames):
            wavfile.write(os.path.join(hparams["train"]["result_dir"], "{}.wav".format(basename)), hparams["audio"]["sampling_rate"], wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="preprocessed_data/DataBaker/test.txt",
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--hparams_path",
        type=str,
        required=True,
        help="path to hparams.yaml",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.text is not None

    # Read Config
    hparams = yaml.load(open(args.hparams_path, "r"), Loader=yaml.FullLoader)

    # Prepare device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Load vocoder
    vocoder = get_vocoder(hparams, device)

    # Load Tacotron Model
    print("\nLoading Tacotron Model...\n")
    model, _ = create_model(hparams)
    model = load_checkpoint(model, os.path.join(hparams["train"]["output_dir"], "checkpoint_{}".format(args.restore_step)))
    model = model.to(device)
    model.eval()

    if args.mode == "batch":
        # Get dataset
        testset = TextDataset(args.source, hparams)
        collate_fn = TextCollate()
        testdata_loader = DataLoader(testset, sampler=None, num_workers=1,
                                     shuffle=False, batch_size=1,
                                     pin_memory=False, collate_fn=collate_fn)
        for i, batch in enumerate(testdata_loader):
            synthesize_batch(hparams, model, batch)

    if args.mode == "single":
        basenames = [args.text]
        if hparams["lang"] == "en":
            phone_text = raw_text = args.text.lower()
            print("Raw Text: ", raw_text)
            nlp = stanza.Pipeline("en", "/apdcephfs/private_yatsenzhou/pretrained/stanza/stanza_en")
            bert = BertModel.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-uncased")
            tokenizer = BertTokenizer.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-uncased/bert-base-uncased-vocab.txt")
            phone, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id = get_text(tokenizer, bert, nlp, "en", phone_text, raw_text)

        elif hparams["lang"] == "zh":
            raw_text = args.text
            phone_text = " ".join([p[0] for p in pinyin(raw_text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)])
            phone_text = phone_text.replace("，", ",")
            phone_text = phone_text.replace("。", ".")
            phone_text = phone_text.replace("！", "!")
            phone_text = phone_text.replace("？", "?")
            print("Raw Text: ", raw_text)
            print("Pinyin Sequence: ", phone_text)
            nlp = stanza.Pipeline("zh", "/apdcephfs/private_yatsenzhou/pretrained/stanza/stanza_zh")
            bert = BertModel.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-chinese")
            tokenizer = BertTokenizer.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-chinese/bert-base-chinese-vocab.txt")
            phone, word_emb, phone2word_idx, nodes_list1, nodes_list2, deprels_id = get_text(tokenizer, bert, nlp, "zh", phone_text, raw_text)

        text = torch.LongTensor([text_to_sequence(phone, hparams["text_cleaners"])])
        text_length = torch.LongTensor([len(text[0])])
        word_emb = torch.unsqueeze(word_emb, 0)
        phone2word_idx = torch.LongTensor([phone2word_idx])
        g = dgl.graph((nodes_list1, nodes_list2), num_nodes=len(word_emb[0]))
        g.edata['type'] = torch.LongTensor(deprels_id)
        g_batch = dgl.batch([g])
        batch = (text, text_length, word_emb, phone2word_idx, g_batch), basenames
        synthesize_batch(hparams, model, batch)


