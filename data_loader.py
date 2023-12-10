import os
import numpy as np 
import torch
import logging
import copy
import json

from transformers import AutoTokenizer
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from utils import get_intent_labels, get_slot_labels
from utils import get_intent_labels, get_slot_labels, get_clean_labels, get_slots_all

#test
logger = logging.getLogger(__name__)

def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        # Tokenize word by word (for NER)
        tokens = []
        heads = []
        # slot_labels_ids = []
        for word, slot_label in zip(example.text, example.slot_labels[1:-1]):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            heads.append(len(tokens) + 1) # +1 for the cls token
            tokens.extend(word_tokens)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        heads += [len(tokens) + 1]
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        heads = [0] + heads
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(heads) == len(example.slot_labels)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("heads: %s" % " ".join([str(x) for x in heads]))
        
        features.append(
            InputExample(guid=example.guid,
                         words=input_ids,
                         chars=example.chars,
                         heads=heads,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         intent_label=example.intent_label,
                         slot_labels=example.slot_labels,
                         text=example.text))

    return features


class Vocab(object):

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2index = {}
        self.index2word = []
        self.special_tokens = ['<PAD>', '<UNK>', '<s>', '</s>']

        self.count = {}

        self.pad_token = '<PAD>'
        self.pad_index = 0
        self.add(self.pad_token)
    
        self.unk_token = '<UNK>'
        self.unk_index = 1
        self.add(self.unk_token)

        self.start_token = '<s>'
        self.start_index = 2
        self.add(self.start_token)

        self.end_token = '</s>'
        self.end_index = 3
        self.add(self.end_token)

    def add(self, token):
        if isinstance(token, (list, tuple)):
            for element in token:
                self.add(element)
            return
        
        assert isinstance(token, str)

        if self.min_freq > 1 and token not in self.special_tokens:
            if len(token) > 1 and not token[0].isalnum():
                token = token[1:]

            if len(token) > 1 and not token[-1].isalnum():
                token = token[:-1]

        if token not in self.count:
            self.count[token] = 0
        self.count[token] += 1

        if token in self.special_tokens or (token not in self.word2index and self.count[token] >= self.min_freq):
            self.word2index[token] = len(self.index2word)
            self.index2word.append(token)
    
    def get_index(self, token):
        if isinstance(token, list):
            return [self.get_index(element) for element in token]
        
        assert isinstance(token, str)

        return self.word2index.get(token, self.unk_index)

    def get_token(self, index):
        if isinstance(index, list):
            return [self.get_token(element) for element in index]
        
        assert isinstance(index, int)
        return self.index2word[index]

    def save(self, path):
        torch.save(self.index2word, path)
    
    def load(self, path):
        self.index2word = torch.load(path)
        self.word2index = {word: i for i, word in enumerate(self.index2word)}

    def __len__(self):
        return len(self.index2word)
    
    def __str__(self):
        return f'Vocab object with {len(self.index2word)} instances'


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, chars=None, heads=None, attention_mask=None, token_type_ids=None, intent_label=None, slot_labels=None, text=None):
        self.guid = guid
        self.words = words
        self.chars = chars
        self.heads = heads
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.text = text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


    
class TextLoader(Dataset):

    def __init__(self, args, mode):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels, self.hiers = get_slots_all(args)

        self.vocab = Vocab(min_freq=self.args.min_freq)
        self.chars = Vocab()
        self.examples = self.build(mode)
    def load_bert(self, tokenizer):
        pad_token_label_id = self.args.ignore_index
        self.examples = convert_examples_to_features(self.examples, self.args.max_seq_len, tokenizer,
                                                     pad_token_label_id=pad_token_label_id)
    @classmethod
    def read_file(cls, input_file, quotechar=None):
        """ Read data file of given path.
        :param file_path: path of data file.
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents = [], [], []
        text, slot = [], []

        with open(input_file, 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    texts.append(text)
                    slots.append(slot)
                    if "/" not in items[0]:
                        intents.append(items)
                    else:
                        new = items[0].split("/")
                        intents.append([new[1]])

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents

    def _create_examples(self, texts, chars, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, char, intent, slot) in enumerate(zip(texts, chars, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = self.vocab.get_index(text)  # Some are spaced twice
            words = [self.vocab.start_index] + words + [self.vocab.end_index]
            # char
            char  = self.chars.get_index(char)
            max_char = max([len(x) for x in char])
            for j in range(len(char)):
                char[j] = char[j] + [0] * (max_char - len(char[j]))
            char = [[0] * max_char] + char + [[0] * max_char]
            # 2. intent
            _intent = intent[0].split('#')
            intent_label = [0 for _ in self.intent_labels]
            for _int in _intent:
                idx = self.intent_labels.index(_int) if _int in self.intent_labels else self.intent_labels.index("UNK")
                intent_label[idx] = 1
            # 3. slot
            slot_labels = []
            for s in slot:
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
            slot_labels = [self.slot_labels.index('PAD')] + slot_labels + [self.slot_labels.index('PAD')]
            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, chars=char, intent_label=intent_label, slot_labels=slot_labels, text=text))
        return examples

    def build(self, mode):
        data_path = os.path.join(self.args.data_dir, self.args.task, mode + '.txt')
        logger.info("LOOKING AT {}".format(data_path))
        texts, slots, intents = self.read_file(data_path)

        chars = []
        max_len = 0
        for text in texts:
            chars.append([])
            for word in text:
                chars[-1].append(list(word))

        cache = os.path.join(self.args.data_dir, f'vocab_{self.args.task}')
        if os.path.exists(cache):
            self.vocab.load(cache)
        elif mode == 'train':
            self.vocab.add(texts)
            self.vocab.save(cache)
        cache_chars = os.path.join(self.args.data_dir, f'chars_{self.args.task}')
        if os.path.exists(cache_chars):
            self.chars.load(cache_chars)
        elif mode == 'train':
            self.chars.add(chars)
            self.chars.save(cache_chars)
        
        return self._create_examples(texts=texts,
                                     chars=chars,
                                     intents=intents,
                                     slots=slots,
                                     set_type=mode)
    
    def __getitem__(self, index):
        example = self.examples[index]
        words = torch.tensor(example.words, dtype=torch.long)
        
        intent = torch.tensor(example.intent_label, dtype=torch.float)
        slot = torch.tensor(example.slot_labels, dtype=torch.long)
        chars = torch.tensor(example.chars, dtype=torch.long)

        if 'bert' in self.args.model_type:
            attention_mask = torch.tensor(example.attention_mask, dtype=torch.long)
            token_type_ids = torch.tensor(example.token_type_ids, dtype=torch.long)
            heads = torch.tensor(example.heads, dtype=torch.long)
            return (words, chars, heads, attention_mask, token_type_ids, intent, slot)
        else:
            return (words, chars, intent, slot)

    def __len__(self):
        return len(self.examples)

class TextCollate():
    def __init__(self, pad_index, num_intents, max_seq_len):
        self.pad_index = pad_index
        self.num_intents = num_intents
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        
        len_list = [len(x[-1]) for x in batch]
        len_char = [x[1].size(1) for x in batch]
        max_len = max(len_list)
        max_char = max(len_char)

        seq_lens = []

        bert = len(batch[0]) > 4
        
        char_padded = torch.LongTensor(len(batch), max_len, max_char)
        slot_padded = torch.LongTensor(len(batch), max_len)
        intent = torch.FloatTensor(len(batch), self.num_intents)
        char_padded.zero_()
        intent.zero_()
        slot_padded.zero_()
        
        if not bert:
            text_padded = torch.LongTensor(len(batch), max_len)
            text_padded.zero_()
            
        else:
            input_ids = torch.LongTensor(len(batch), self.max_seq_len)
            attention_mask = torch.LongTensor(len(batch), self.max_seq_len)
            token_type_ids = torch.LongTensor(len(batch), self.max_seq_len)
            heads = torch.LongTensor(len(batch), max_len)
            input_ids.zero_()
            attention_mask.zero_()
            token_type_ids.zero_()
            heads.zero_()
        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        for i, index in enumerate(sorted_index):
            seq_lens.append(len_list[index])
            intent[i] = batch[index][-2]
            slot = batch[index][-1]
            slot_padded[i, :slot.size(0)] = slot
            char = batch[index][1]
            char_padded[i, :char.size(0), :char.size(1)] = char

            if not bert:
                text = batch[index][0]
                text_padded[i, :text.size(0)] = text
            else:
                input_ids[i] = batch[index][0]
                attention_mask[i] = batch[index][3]
                token_type_ids[i] = batch[index][4]
                head = batch[index][2]
                heads[i, :head.size(0)] = head
        if not bert:
            return text_padded, char_padded, intent, slot_padded, torch.tensor(seq_lens, dtype=torch.long)
        else:
            return input_ids, char_padded, heads, attention_mask, token_type_ids, intent, slot_padded, torch.tensor(seq_lens, dtype=torch.long)
    
if __name__ == '__main__':
    train_dataset = TextLoader(args, 'train')
    print([x.shape for x in train_dataset[0]])
    print([x.shape for x in train_dataset.load_bert()[0]])
