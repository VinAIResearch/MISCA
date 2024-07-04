import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP, get_intent_labels, get_slots_all
from data_loader import TextLoader, TextCollate


def main(args):
    init_logger()
    set_seed(args)
    slot_label_lst, hiers = get_slots_all(args)
    collate = TextCollate(0, len(get_intent_labels(args)), args.max_seq_len)

    train_dataset = TextLoader(args, 'train')
    dev_dataset = TextLoader(args, 'dev')
    test_dataset = TextLoader(args, 'test')

    trainer = Trainer(args, collate, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate('dev', 0)
        trainer.evaluate("test", -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
    parser.add_argument("--slot_label_clean", default="slot_clean.txt", type=str, help="Slot Label file")
    parser.add_argument("--logging", default="log.txt", type=str, help="Logging file")

    # LAAT
    parser.add_argument("--n_levels", default=1, type=int, help="Number of attention")
    parser.add_argument("--attention_mode", default='label', type=str)
    parser.add_argument("--level_projection_size", default=32, type=int)
    parser.add_argument("--d_a", default=-1, type=int)

    parser.add_argument("--char_embed", default=64, type=int)
    parser.add_argument("--char_out", default=64, type=int)
    parser.add_argument("--no_charcnn", action="store_true", help="Whether to use CharCNN")
    parser.add_argument("--no_charlstm", action="store_true", help="Whether to use CharLSTM")
    parser.add_argument("--freeze", action="store_true", help="Whether to use CharLSTM")
    parser.add_argument("--word_embedding_dim", default=128, type=int)
    parser.add_argument("--encoder_hidden_dim", default=128, type=int)
    parser.add_argument("--decoder_hidden_dim", default=256, type=int)
    parser.add_argument("--attention_hidden_dim", default=256, type=int)
    parser.add_argument("--attention_output_dim", default=256, type=int)

    # Config training
    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=100, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=50, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=-1, help="Log every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--tuning_metric", default="mean_intent_slot", type=str, help="Metric to save checkpoint")

    parser.add_argument("--only_intent", default=0, type=float, help="The first epochs to optimize intent")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )

    parser.add_argument('--intent_loss_coef', type=float, default=0.5, help='Coefficient for the intent loss.')
    parser.add_argument('--aux_loss_coef', type=float, default=0.5, help='Coefficient for the aux task.')
    parser.add_argument('--early_stopping', type=float, default=-1, help='Early stopping strategy')

    parser.add_argument("--base_model", default=None, type=str, help="The pretrained model path")

    parser.add_argument(
        "--num_intent_detection",
        action="store_true",
        help="Whether to use two-stage intent detection",
    )

    parser.add_argument(
        "--auxiliary_tasks",
        action="store_true",
        help="Whether to optimize with auxiliary tasks",
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

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
