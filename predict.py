import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP, get_intent_labels, get_slots_all
from data_loader import TextLoader, TextCollate, Vocab

def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines

def load_model(args):
    slot_label_lst, hiers = get_slots_all(args)
    intent_label_lst = get_intent_labels(args)
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    if 'bert' in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        config = config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            args=args,
            intent_label_lst=intent_label_lst,
            slot_label_lst=slot_label_lst,
            slot_hier=hiers
        )
    else:
        vocab = Vocab(min_freq=args.min_freq)
        chars = Vocab()
        f_voc = os.path.join(args.data_dir, f'vocab_{args.task}')
        vocab.load(f_voc)
        f_chr = os.path.join(args.data_dir, f'chars_{args.task}')
        chars.load(f_chr)
        args.n_chars = len(chars)
        model = model_class(args, len(vocab), intent_label_lst, slot_label_lst, hiers)

    pretrained_state = torch.load(os.path.join(args.model_dir, 'model.bin'))
    model.load_state_dict(pretrained_state)
    return model

def data_lstm(args, data, vocab, char_voc, max_len, max_char):
    all_words = []
    all_chars = []
    seq_lens = []
    chars = []
    for text in data:
        chars.append([])
        for word in text:
            chars[-1].append(list(word))
    for line, char in zip(data, chars):
        tokens = []
        words = vocab.get_index(line)
        words = [vocab.start_index] + words + [vocab.end_index]
        seq_lens.append(len(words))
        words = words + [0] * (max_len - len(words))
        chrs  = char_voc.get_index(char)
        for j in range(len(char)):
            chrs[j] = chrs[j] + [0] * (max_char - len(chrs[j]))
        chrs = [[0] * max_char] + chrs + [[0] * max_char]
        chrs = chrs + [[0] * max_char] * (max_len - len(chrs))
        all_words.append(words)
        all_chars.append(chrs)
    all_words = torch.tensor(all_words, dtype=torch.long)
    all_chars = torch.tensor(all_chars, dtype=torch.long)
    seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    dataset = TensorDataset(all_words, all_chars, seq_lens)
    return dataset

def data_bert(args, data, tokenizer, max_seq_len,
              pad_token_label_id=-100,
              cls_token_segment_id=0,
              pad_token_segment_id=0,
              sequence_a_segment_id=0,
              mask_padding_with_zero=True):
    
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    seq_lens = []
    all_heads = []
    num_tokens = []
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    for words in data:
        tokens = []
        slot_label_mask = []
        heads = []
        num_tokens.append(len(words))
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            heads.append(len(tokens) + 1)
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        heads += [len(tokens) + 1]
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        heads += [0] + heads
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        seq_lens.append(len(input_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        heads = heads + [0] * (args.max_seq_len - len(heads))
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_heads.append(heads)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_heads = torch.tensor(all_heads, dtype=torch.long)
    seq_lens =  torch.tensor(seq_lens, dtype=torch.long)
    all_tokens = torch.tensor(num_tokens, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_heads, seq_lens, all_tokens)

    return dataset

def main(args):
    slot_label_lst, hiers = get_slots_all(args)
    intent_lst = get_intent_labels(args)
    model = load_model(args)
    data = read_input_file(args)
    if 'bert' in args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        dataset = data_bert(args, data, tokenizer, args.max_seq_len)
    else:
        vocab = Vocab(min_freq=args.min_freq)
        chars = Vocab()
        f_voc = os.path.join(args.data_dir, f'vocab_{args.task}')
        vocab.load(f_voc)
        f_chr = os.path.join(args.data_dir, f'chars_{args.task}')
        chars.load(f_chr)
        dataset = data_lstm(args, data, vocab, chars, 100, 50)
    predict(args, model, data, dataset, intent_lst, slot_label_lst)
    
def predict(args, model, text, dataset, intent_lst, slot_label_lst, pad_token_label_id=-100):
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    intent_label_map = {i: label for i, label in enumerate(intent_lst)}
    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None
    
    all_intent = []
    all_slot = []

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                    "attention_mask": batch[1],
                    "intent_label_ids": None,
                    "slot_labels_ids":None,
                    "token_type_ids": batch[2],
                    "heads": batch[3],
                    "seq_lens": batch[4].cpu()
                    }
            outputs = model(**inputs)
            intent_logits, slot_logits, num_intent = outputs[1]
            
            intent_logits = F.logsigmoid(intent_logits).detach().cpu()
            intent_preds = intent_logits.numpy()
            intent_nums = num_intent.detach().cpu().numpy()
            if args.use_crf:
                slot_preds = np.array(model.crf.decode(slot_logits))
            else:
                slot_preds = slot_logits.detach().cpu()
                
            intent_nums = np.argmax(intent_nums, axis=-1)
            for num, preds in zip(intent_nums, intent_preds):
                idx = preds.argsort()[-num:]
                it = list(map(lambda x : intent_label_map[x], sorted(idx)))
                all_intent.append('#'.join(it))
            if not args.use_crf:
                slot_preds_arg = np.argmax(slot_preds.numpy(), axis=2)
            else:
                slot_preds_arg = slot_preds
                
            for i in range(slot_preds_arg.shape[0]):
                all_slot.append([])
                for j in range(batch[5][i]):
                    all_slot[-1].append(slot_label_map[slot_preds_arg[i][j]])

    # Write to output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(text, all_slot, all_intent):
            line = ""
            assert len(words) == len(slot_preds)
            slt = None
            mention = ''
            for word, pred in zip(words, slot_preds):
                if pred[:2] == 'B-':
                    if slt:
                        line = line + "[{}:{}] ".format(mention, slt)
                    slt = pred[2:]
                    mention = word
                elif pred[:2] == 'I-':
                    mention = mention + " " + word
                else:
                    if slt:
                        line = line + "[{}:{}] ".format(mention, slt)
                    line = line + word + " "
                    slt = None
                    mention = ''
            if slt:
                line = line + "[{}:{}] ".format(mention, slt)
                
            f.write("<{}> -> {}\n".format(intent_pred, line.strip()))

    print("Prediction Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
    parser.add_argument("--slot_label_clean", default="slot_clean.txt", type=str, help="Slot Label file")

    # LAAT
    parser.add_argument("--n_levels", default=1, type=int, help="Number of attention")
    parser.add_argument("--attention_mode", default=None, type=str)
    parser.add_argument("--level_projection_size", default=32, type=int)
    parser.add_argument("--d_a", default=-1, type=int)

    parser.add_argument("--char_embed", default=64, type=int)
    parser.add_argument("--char_out", default=64, type=int)
    parser.add_argument("--no_charcnn", action="store_true", help="Whether to use CharCNN")
    parser.add_argument("--no_charlstm", action="store_true", help="Whether to use CharLSTM")
    parser.add_argument("--word_embedding_dim", default=128, type=int)
    parser.add_argument("--encoder_hidden_dim", default=128, type=int)
    parser.add_argument("--decoder_hidden_dim", default=256, type=int)
    parser.add_argument("--attention_hidden_dim", default=256, type=int)
    parser.add_argument("--attention_output_dim", default=256, type=int)

    # Config training
    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=100, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )


    parser.add_argument(
        "--num_intent_detection",
        action="store_true",
        help="Whether to use two-stage intent detection",
    )
    
    parser.add_argument(
        "--slot_decoder_size", type=int, default=512, help="hidden size of attention output vector"
    )

    parser.add_argument(
        "--intent_slot_attn_size", type=int, default=256, help="hidden size of attention output vector"
    )

    parser.add_argument(
        "--min_freq", type=int, default=1, help="Minimum number of frequency to be considered in the vocab"
    )

    parser.add_argument(
        '--intent_slot_attn_type', choices=['coattention', 'attention_flow'], 
    )

    parser.add_argument(
        '--embedding_type', choices=['soft', 'hard'], default='soft', 
    )

    parser.add_argument(
        "--label_embedding_size", type=int, default=256, help="hidden size of label embedding vector"
    )

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    parser.add_argument("--input_file", default="input.txt", type=str, help="File input")
    parser.add_argument("--output_file", default="output.txt", type=str, help="File input")
    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
