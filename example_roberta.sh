# Train the base model
python main.py  --model_type roberta \
                --model_dir dir_base \
                --task <mixatis or mixsnips> \
                --data_dir data \
                --do_train \
                --do_eval \
                --num_train_epochs 40 \
                --intent_loss_coef <lambda> \
                --learning_rate 1e-5 \
                --num_intent_detection \
                --use_crf

# Train MISCA with encoder freezed
python main.py --model_type roberta \
            --model_dir freeze \
            --task <mixatis or mixsnips> \
            --data_dir data \
            --do_train \
            --do_eval \
            --num_train_epochs 40 \
            --intent_loss_coef <lambda> \
            --learning_rate 1e-5 \
            --num_intent_detection \
            --use_crf \
            --base_model dir_base \
            --intent_slot_attn_type coattention \
            --freeze

# Fine tune the whole model
python main.py --model_type roberta \
            --model_dir misca \
            --task <mixatis or mixsnips> \
            --data_dir data \
            --do_train \
            --do_eval \
            --num_train_epochs 20 \
            --intent_loss_coef <lambda> \
            --learning_rate 1e-5 \
            --num_intent_detection \
            --use_crf \
            --base_model freeze \
            --intent_slot_attn_type coattention