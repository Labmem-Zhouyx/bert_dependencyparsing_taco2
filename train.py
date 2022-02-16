import os
import time
import argparse
import json
import yaml

import torch
from torch.utils.data import DataLoader

import stanza
from pytorch_pretrained_bert import BertModel, BertTokenizer
from model.tacotron import Tacotron, TacotronLoss
from model.tacotron2 import Tacotron2, Tacotron2Loss
from model.semantic_tacotron2 import Semantic_Tacotron2, Semantic_Tacotron2Loss
from utils.dataset import TextMelDataset, TextMelCollate
from utils.logger import TacotronLogger
from utils.utils import data_parallel_workaround
from text import symbols
from vocoder import get_vocoder, vocoder_infer


def prepare_datasets(hparams):
    if hparams["lang"] == "en":
        # stanza.download("en", "path")
        nlp = stanza.Pipeline("en", "/apdcephfs/private_yatsenzhou/pretrained/stanza/stanza_en")
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        bert = BertModel.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-uncased/bert-base-uncased-vocab.txt")
    else:
        # stanza.download("zh", "path")
        nlp = stanza.Pipeline("zh", "/apdcephfs/private_yatsenzhou/pretrained/stanza/stanza_zh")
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
        # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
        bert = BertModel.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-chinese")
        tokenizer = BertTokenizer.from_pretrained("/apdcephfs/private_yatsenzhou/pretrained/bert/bert-base-chinese/bert-base-chinese-vocab.txt")
    # Get data, data loaders and collate function ready
    trainset = TextMelDataset(hparams["dataset"]["training_files"], hparams, nlp, bert, tokenizer)
    valset = TextMelDataset(hparams["dataset"]["validation_files"], hparams, nlp, bert, tokenizer)
    collate_fn = TextMelCollate(hparams["model"]["r"])
    #
    return trainset, valset, collate_fn


def create_model(hparams):
    # Model config
    with open(hparams["model"]["tacotron_config"], 'r') as f:
        model_cfg = json.load(f)
    num_symbols = len(symbols)
    if hparams["model"]["tacotron_version"] == "1":
        # Tacotron model
        model = Tacotron(n_vocab=num_symbols,
                         embed_dim=hparams["model"]["symbols_embed_dim"],
                         mel_dim=hparams["model"]["mel_dim"],
                         linear_dim=hparams["model"]["mel_dim"],
                         max_decoder_steps=hparams["model"]["max_decoder_steps"],
                         stop_threshold=hparams["model"]["stop_threshold"],
                         r=hparams["model"]["r"],
                         model_cfg=model_cfg
                         )
        # Loss criterion
        criterion = TacotronLoss()
    elif hparams["model"]["tacotron_version"] == "2":
        # Tacotron2 model
        model = Semantic_Tacotron2(n_vocab=num_symbols,
                        embed_dim=hparams["model"]["symbols_embed_dim"],
                        mel_dim=hparams["model"]["mel_dim"],
                        max_decoder_steps=hparams["model"]["max_decoder_steps"],
                        stop_threshold=hparams["model"]["stop_threshold"],
                        r=hparams["model"]["r"],
                        use_bert=hparams["model"]["use_bert"],
                        bert_dim=hparams["model"]["bert_dim"],
                        use_dependency=hparams["model"]["use_dependency"],
                        graph_type=hparams["model"]["graph_type"],
                        model_cfg=model_cfg
                        )
        # Loss criterion
        criterion = Semantic_Tacotron2Loss()
        # model = Tacotron2(n_vocab=num_symbols,
        #                 embed_dim=hparams["model"]["symbols_embed_dim"],
        #                 mel_dim=hparams["model"]["mel_dim"],
        #                 max_decoder_steps=hparams["model"]["max_decoder_steps"],
        #                 stop_threshold=hparams["model"]["stop_threshold"],
        #                 r=hparams["model"]["r"],
        #                 model_cfg=model_cfg
        #                 )
        # # Loss criterion
        # criterion = Tacotron2Loss()
    else:
        raise ValueError("Unsupported Tacotron version: {} ".format(hparams["model"]["tacotron_version"]))
    #
    return model, criterion


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    iteration = checkpoint['iteration']

    print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(checkpoint_path, model, optimizer, learning_rate, iteration):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def validate(model, criterion, iteration, device, valset, hparams, collate_fn, logger, vocoder):
    """Evaluate on validation set, get validation loss and printing
    """
    model.eval()
    with torch.no_grad():
        valdata_loader = DataLoader(valset, sampler=None, num_workers=1,
                                shuffle=False, batch_size=hparams["train"]["batch_size"],
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(valdata_loader):
            inputs, targets = model.parse_data_batch(batch)
            predicts = model(inputs)

            # Loss
            loss = criterion(predicts, targets)

            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    wav_reconstructions = vocoder_infer(targets[0].transpose(1, 2), vocoder, hparams, lengths=inputs[-1] * hparams["audio"]["hop_length"])
    wav_predictions = vocoder_infer(predicts[1].transpose(1, 2), vocoder, hparams, lengths=inputs[-1] * hparams["audio"]["hop_length"])

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))

    logger.log_validation(val_loss, model, targets, predicts, wav_reconstructions, wav_predictions, iteration)


def train(hparams, warm_start):
    """Training and validation logging results to tensorboard and stdout
    """

    torch.manual_seed(hparams["train"]["seed"])
    torch.cuda.manual_seed(hparams["train"]["seed"])

    # Prepare device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Decide parallel running on GPU
    parallel_run = hparams["train"]["distributed_run"] and torch.cuda.device_count() > 1

    # Load vocoder
    vocoder = get_vocoder(hparams, device)

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
    model, criterion = create_model(hparams)
    model = model.to(device)

    # Initialize the optimizer
    learning_rate = hparams["optimization"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams["optimization"]["weight_decay"])

    # Prepare directory and logger
    output_dir = hparams["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = TacotronLogger(hparams)

    # Prepare dataset and dataloader
    trainset, valset, collate_fn = prepare_datasets(hparams)
    train_loader = DataLoader(trainset, sampler=None, num_workers=1,
                              shuffle=True, batch_size=hparams["train"]["batch_size"],
                              pin_memory=False, drop_last=True, collate_fn=collate_fn)

    # Load checkpoint if one exists
    iteration    = -1 # will add 1 in main loop
    epoch_offset = 0
    if hparams["train"]["checkpoint_path"] != "":
        if warm_start:
            model = warm_start_model(hparams["train"]["checkpoint_path"], model, hparams["train"]["ignore_layers"])
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(hparams["train"]["checkpoint_path"], model, optimizer)
            if hparams["optimization"]["use_saved_learning_rate"]:
                learning_rate = _learning_rate
            epoch_offset = max(0, int(iteration / len(train_loader)))

    # ================ MAIN TRAINNIG LOOP! ===================
    model.train()
    for epoch in range(epoch_offset, hparams["train"]["epochs"]):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            iteration += 1

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # prepare data
            inputs, targets = model.parse_data_batch(batch)

            # Forward pass
            # Parallelize model onto GPUS using workaround due to python bug
            if parallel_run:
                predicts = data_parallel_workaround(model, inputs)
            else:
                predicts = model(inputs)

            # Loss
            loss = criterion(predicts, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["optimization"]["grad_clip_thresh"])
            optimizer.step()

            # Logs
            duration = time.perf_counter() - start
            print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(iteration, loss, grad_norm, duration))
            logger.log_training(loss, grad_norm, learning_rate, duration, iteration)

            # Validation
            if iteration % hparams["train"]["iters_per_validation"] == 0:
                validate(model, criterion, iteration, device, valset, hparams, collate_fn, logger, vocoder)

            # Save checkpoint
            if iteration % hparams["train"]["iters_per_checkpoint"] == 0:
                checkpoint_path = os.path.join(output_dir, "checkpoint_{}".format(iteration))
                save_checkpoint(checkpoint_path, model, optimizer, learning_rate, iteration)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_path', type=str, required=True, help='path to hparams.yaml')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')

    args = parser.parse_args()
    hparams = yaml.load(open(args.hparams_path, "r"), Loader=yaml.FullLoader)

    torch.backends.cudnn.enabled = hparams["train"]["cudnn_enabled"]
    torch.backends.cudnn.benchmark = hparams["train"]["cudnn_benchmark"]

    print("Dynamic Loss Scaling:", hparams["train"]["dynamic_loss_scaling"])
    print("Distributed Run:", hparams["train"]["distributed_run"])
    print("cuDNN Enabled:", hparams["train"]["cudnn_enabled"])
    print("cuDNN Benchmark:", hparams["train"]["cudnn_benchmark"])

    train(hparams, args.warm_start)
