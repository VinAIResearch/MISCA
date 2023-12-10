import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve

from transformers import RobertaConfig, RobertaTokenizer
from model import JointLSTM, JointRoberta

MODEL_CLASSES = {
    "lstm": (None, JointLSTM, None),
    "roberta": (RobertaConfig, JointRoberta, RobertaTokenizer)
}

MODEL_PATH_MAP = {
    "lstm": "",
    "roberta": "roberta-base"
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

def get_clean_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_clean), 'r', encoding='utf-8')]

def get_slots_all(args):
    slot_labels = get_slot_labels(args)
    hier = ()
    if args.task == 'mixatis':
        slot_parents = get_clean_labels(args)
        hier = (slot_parents, )
    slot_type = sorted(set([name[2:] for name in slot_labels if name[:2] == 'B-' or name[:2] == 'I-']))
    hier += (slot_type, )
    return slot_labels, hier



def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    # print(len(intent_preds), len(intent_labels), len(slot_preds), len(slot_labels))
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    mean_intent_slot = (intent_result["intent_acc"] + slot_result["slot_f1"]) / 2

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)
    results["mean_intent_slot"] = mean_intent_slot

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    # average_precision = average_precision_score(labels.reshape(-1), preds.reshape(-1))
    acc = ((preds == labels).all(1)).mean()

    tp = preds == 1.
    tl = labels == 1.
    correct = np.multiply(tp, tl).sum()

    tp = np.sum(tp)
    tl = np.sum(tl)

    p = correct / tp if tp > 0 else 0.0
    r = correct / tl if tl > 0 else 0.0
    f1 = 0.0 if p + r == 0.0 else 2 * p * r / (p + r)

    return {
        "intent_acc": acc,
        "intent_f1": f1,
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels).all(1)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)
    slot_acc = slot_result.mean()

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "semantic_frame_acc": sementic_acc,
        "slot_acc": slot_acc
    }
