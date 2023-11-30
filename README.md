# MISCA: A Joint Model for Multiple Intent Detection and Slot Filling with Intent-Slot Co-Attention

- We propose a joint model (namely, MISCA) for multi-intent detection and slot filling, which incorporates label attention and intent-slot co-attention mechanisms.
- Experiments on MixATIS and MixSNIPS datasets show that our proposed model achieved state-of-the-art results.

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

### Acknowledgement
Our code is based on the implementation of the JointIDSF paper from https://github.com/VinAIResearch/JointIDSF
