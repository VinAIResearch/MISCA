# MISCA: A Joint Model for Multiple Intent Detection and Slot Filling with Intent-Slot Co-Attention

- We propose a joint model (namely, MISCA) for multi-intent detection and slot filling, which incorporates label attention and intent-slot co-attention mechanisms.
- Experiments on MixATIS and MixSNIPS datasets show that our proposed model achieved state-of-the-art results.

<p align="center">	
<img width="600" alt="model" src="model.png">
</p>

**Please CITE** our paper whenever our dataset or model implementation is used to help produce published results or incorporated into other software.

    @inproceedings{MISCA,
        title     = {{MISCA: A Joint Model for Multiple Intent Detection and Slot Filling with Intent-Slot Co-Attention}},
        author    = {Thinh Pham and Chi Tran and Dat Quoc Nguyen},
        booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
        year      = {2023}
    }


## Model installation, training and evaluation

### Installation
- Python version >= 3.8
- PyTorch version >= 1.8.0

```
    git clone https://github.com/VinAIResearch/MISCA.git
    cd MISCA/
    pip3 install -r requirements.txt
```

### Training and evaluation
You can run the experiments by the following command.
```
python main.py --token_level word-level \
            --model_type lstm \
            --model_dir <dir to save model> \
            --task <name of dataset, mixatis or mixsnips> \
            --data_dir <dir of data> \
            --seed 1 \
            --attention_mode label \
            --do_train \
            --do_eval \
            --num_train_epochs 100 \
            --intent_loss_coef <lambda> \
            --learning_rate 1e-5 \
            --aux_loss_coef <lambda> \
            --train_batch_size 32 \
            --char_embed 64 \
            --char_out 64 \
            --word_embedding_dim 128 \
            --encoder_hidden_dim 128 \
            --attention_hidden_dim 256 \
            --attention_output_dim 256 \
            --decoder_hidden_dim 256 \
            --use_charcnn \
            --use_charlstm \
            --num_intent_detection \
            --use_crf \
            --slot_decoder_size 128 \
            --level_projection_size 32 \
            --pretrained \
            --pretrained_path $PRETRAINED_PATH \
            --intent_slot_attn_size 128 \
            --intent_slot_attn_type coattention \
            --embedding_type soft \
            --label_embedding_size 128
```
It is noted that we train the base model without coattention first and initialize MISCA with this base model. To train the base model, simply remove the last 6 lines in the command above. 

To train the model with PLMs, users must replace the token embeddings with the first subword representation from PLMs.

Due to some stochastic factors(e.g., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results. The suggested settings are provided in our paper.

If you have any question, please issue the project or email me (v.thinhphp1@vinai.io or thinhphp.nlp@gmail.com) and we will reply you soon.
### Acknowledgement
Our code is based on the implementation of the JointIDSF paper from https://github.com/VinAIResearch/JointIDSF
