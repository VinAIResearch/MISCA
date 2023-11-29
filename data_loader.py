import os
import numpy as np 
import torch
import logging
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import get_intent_labels, get_slot_labels, get_clean_labels, get_slots_all

logger = logging.getLogger(__name__)

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

    def __init__(self, guid, words, chars, intent_label=None, slot_labels=None, slot_hiers=None):
        self.guid = guid
        self.words = words
        self.chars = chars
        self.intent_label = intent_label
        self.slot_labels = slot_labels
        self.slot_hiers = slot_hiers

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

        self.vocab = Vocab(min_freq=args.min_freq)
        self.chars = Vocab()
        self.examples = self.build(mode)

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
            # print(char)
            # 2. intent
            _intent = intent[0].split('#')
            intent_label = [0 for _ in self.intent_labels]
            for _int in _intent:
                idx = self.intent_labels.index(_int) if _int in self.intent_labels else self.intent_labels.index("UNK")
                intent_label[idx] = 1
            # 3. slot
            slot_labels = []
            layers = [[0 for _ in x] for x in self.hiers]
            for s in slot:
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
                if s[:2] == 'B-':
                    name = s[2:]
                    lays = name.split('.')
                    for l in range(len(lays)):
                        pref = '.'.join(lays[:l + 1])
                        if pref in self.hiers[l]:
                            idx = self.hiers[l].index(pref)
                            layers[l][idx] = 1
            slot_labels = [self.slot_labels.index('PAD')] + slot_labels + [self.slot_labels.index('PAD')]
            
            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, chars=char, intent_label=intent_label, slot_labels=slot_labels, slot_hiers=layers))
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

        cache = os.path.join(self.args.data_dir, f'vocab_{self.args.task}_{self.args.min_freq}')
        if os.path.exists(cache):
            self.vocab.load(cache)
        elif mode == 'train':
            # print(texts)
            self.vocab.add(texts)
            self.vocab.save(cache)
        cache_chars = os.path.join(self.args.data_dir, f'chars_{self.args.task}')
        if os.path.exists(cache_chars):
            self.chars.load(cache_chars)
        elif mode == 'train':
            # print(texts)
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
        chars = torch.tensor(example.chars, dtype=torch.long)
        intent = torch.tensor(example.intent_label, dtype=torch.float)
        slot = torch.tensor(example.slot_labels, dtype=torch.long)
        hiers = torch.tensor([it for lst in example.slot_hiers for it in lst], dtype=torch.float)
        return (words, chars, intent, slot, hiers)

    def __len__(self):
        return len(self.examples)

class TextCollate():
    def __init__(self, pad_index, num_intents, hiers):
        self.pad_index = pad_index
        self.num_intents = num_intents
        self.sum_hiers = sum([len(x) for x in hiers])

    def __call__(self, batch):
        
        len_list = [len(x[0]) for x in batch]
        len_char = [x[1].size(1) for x in batch]
        max_len = max(len_list)
        max_char = max(len_char)

        seq_lens = []

        text_padded = torch.LongTensor(len(batch), max_len)
        char_padded = torch.LongTensor(len(batch), max_len, max_char)
        slot_padded = torch.LongTensor(len(batch), max_len)
        intent = torch.FloatTensor(len(batch), self.num_intents)
        slot_hier = torch.FloatTensor(len(batch), self.sum_hiers)
        text_padded.zero_()
        char_padded.zero_()
        slot_padded.zero_()
        intent.zero_()
        slot_hier.zero_()

        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        for i, index in enumerate(sorted_index):
            seq_lens.append(len_list[index])
            text = batch[index][0]
            text_padded[i, :text.size(0)] = text
            char = batch[index][1]
            char_padded[i, :char.size(0), :char.size(1)] = char

            slot = batch[index][3]
            slot_padded[i, :slot.size(0)] = slot

            intent[i] = batch[index][2]
            slot_hier[i] = batch[index][4]
        return text_padded, char_padded, intent, slot_padded, slot_hier, torch.tensor(seq_lens, dtype=torch.long)
    
