import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels, get_clean_labels, get_slots_all

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, collate, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate
        args.n_chars = len(self.train_dataset.chars)
        if 'bert' in self.args.model_type:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            train_dataset.load_bert(self.tokenizer)
            dev_dataset.load_bert(self.tokenizer)
            test_dataset.load_bert(self.tokenizer)

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst, self.hiers = get_slots_all(args)

        self.pad_token_label_id = args.ignore_index
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        if 'bert' in self.args.model_type:
            self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
                slot_hier=self.hiers
            )
        else:
            self.model = self.model_class(args, len(self.train_dataset.vocab), self.intent_label_lst, self.slot_label_lst, self.hiers)
        if args.pretrained:
            model_state = self.model.state_dict()
            pretrained_state = torch.load(os.path.join(args.pretrained_path, 'model.bin'))
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            self.model.load_state_dict(model_state)
            
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, collate_fn=self.collate_fn)
        
        writer = SummaryWriter(log_dir=self.args.model_dir)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        print("check init")
        results = self.evaluate("dev", -1)
        print(results)
        logfile = open(self.args.model_dir + "/" + self.args.logging, 'w')
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        if self.args.logging_steps < 0:
            self.args.logging_steps = len(train_dataloader)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        best_sent = 0
        best_slot = 0

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            print("\nEpoch", _)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch[:-1]) + (batch[-1], ) # GPU or CPU
                if 'bert' in self.args.model_type:
                       inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[3],
                    "intent_label_ids": batch[5],
                    "slot_labels_ids": batch[6],
                    "token_type_ids": batch[4],
                    "heads": batch[2],
                    "seq_lens": batch[-1].cpu()
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "char_ids": batch[1],
                        "intent_label_ids": batch[2],
                        "slot_labels_ids": batch[3],
                        "seq_lens": batch[4],
                    }
                outputs = self.model(**inputs)
                total_loss, intent_loss, slot_loss, count_loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.args.gradient_accumulation_steps
                if _ < self.args.num_train_epochs * self.args.only_intent:
                    total_loss = intent_loss + count_loss 
                    total_loss.backward()
                else:
                    total_loss.backward()

                tr_loss += total_loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % (self.args.logging_steps) == 0:
                        print("\nTuning metrics:", self.args.tuning_metric)
                        results = self.evaluate("dev", _)
                        # self.evaluate("test")
                        writer.add_scalar("Loss/validation", results["loss"], _)
                        writer.add_scalar("Intent Accuracy/validation", results["intent_acc"], _)
                        writer.add_scalar("Intent F1", results["intent_f1"], _)
                        writer.add_scalar("Slot F1/validation", results["slot_f1"], _)
                        writer.add_scalar("Mean Intent Slot", results["mean_intent_slot"], _)
                        writer.add_scalar("Sentence Accuracy/validation", results["semantic_frame_acc"], _)

                        if results['semantic_frame_acc'] >= best_sent or results['slot_f1'] >= best_slot:
                            best_sent = results['semantic_frame_acc']
                            best_slot = results['slot_f1']
                            self.save_model()
                            results = self.evaluate('test', _)
                            logfile.write('\n\nEPOCH = ' + str(_) + '\n')
                            for key in sorted(results.keys()):
                                to_write = " {key} = {value}".format(key=key, value=str(results[key]))
                                logfile.write(to_write)
                                logfile.write("\n")

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
            writer.add_scalar("Loss/train", tr_loss / global_step, _)
        logfile.close()
        return global_step, tr_loss / global_step

    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode, epoch):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, collate_fn=self.collate_fn)

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = []
        slot_preds_list = []
        predictions = []
        intent_labels = []
        int_len_gold = []
        int_len_pred = []

        results = {}
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch[:-1]) + (batch[-1], )
            # print(batch)
            with torch.no_grad():
                if 'bert' in self.args.model_type:
                       inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[3],
                    "intent_label_ids": batch[5],
                    "slot_labels_ids": batch[6],
                    "token_type_ids": batch[4],
                    "heads": batch[2],
                    "seq_lens": batch[-1].cpu()
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "char_ids": batch[1],
                        "intent_label_ids": batch[2],
                        "slot_labels_ids": batch[3],
                        "seq_lens": batch[4],
                    }
                outputs = self.model(**inputs)
                
                if self.args.num_intent_detection:
                    tmp_eval_loss, (intent_logits, slot_logits, intent_dec) = outputs[:2]
                else:
                    tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss[0].mean().item()
            nb_eval_steps += 1

            # Intent prediction
            intent_logits = F.logsigmoid(intent_logits).detach().cpu()
            intent_preds = intent_logits.numpy()
            if self.args.num_intent_detection:
                intent_nums = intent_dec.detach().cpu().numpy()
            out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
            intent_labels.extend(out_intent_label_ids.tolist())

            # Slot prediction
            
            if self.args.use_crf:
                slot_preds = np.array(self.model.crf.decode(slot_logits))
            else:
                slot_preds = slot_logits.detach().cpu()
            out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()

            cur = []
            if self.args.num_intent_detection:
                num_intents = intent_logits.size(1)
                intent_nums = np.argmax(intent_nums, axis=-1)
                gold_nums = np.sum(out_intent_label_ids, axis=-1)
                int_len_gold.extend(gold_nums.tolist())
                int_len_pred.extend(intent_nums.tolist())
                for num, preds in zip(intent_nums, intent_preds):
                    idx = preds.argsort()[-num:]
                    p = np.zeros(num_intents)
                    p[idx] = 1.
                    predictions.append(p)
                    cur.append(p)
            else:
                predictions.extend(np.rint(intent_preds).tolist())

            if not self.args.use_crf:
                slot_preds_arg = np.argmax(slot_preds.numpy(), axis=2)
            else:
                slot_preds_arg = slot_preds
            
            for i in range(out_slot_labels_ids.shape[0]):
                slt = None
                out_slot_label_list.append([])
                slot_preds_list.append([])
                for j in range(out_slot_labels_ids.shape[1]):
                    if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                        out_slot_label_list[-1].append(slot_label_map[out_slot_labels_ids[i][j]])
                        
                        predict_label = slot_label_map[slot_preds_arg[i][j]]
                        if predict_label[:2] == 'B-':
                            slt = predict_label[2:]
                        elif predict_label[:2] == 'I-':
                            if slt is None:
                                predict_label = 'O'
                            elif slt != predict_label[2:]:
                                predict_label = 'O'
                        else:
                            slt = None
                        slot_preds_list[-1].append(predict_label)
        eval_loss = eval_loss / nb_eval_steps
        results['loss'] = eval_loss
        predictions = np.array(predictions)
        intent_labels = np.array(intent_labels)
        total_result = compute_metrics(predictions, intent_labels, slot_preds_list, out_slot_label_list)
        results.update(total_result)
        int_len_gold = np.array(int_len_gold)
        int_len_pred = np.array(int_len_pred)
        results['num_acc'] = (int_len_gold == int_len_pred).mean()
        results['epoch'] = epoch
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save, os.path.join(self.args.model_dir, 'model.bin'))

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'model.bin')), strict=False)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")