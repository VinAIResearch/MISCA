# Train the base model
python main.py  --model_type lstm \
                --model_dir dir_base \
                --task <mixatis or mixsnips> \
                --data_dir data \
                --do_train \
                --do_eval \
                --num_train_epochs 100 \
                --intent_loss_coef <lambda> \
                --learning_rate 1e-3 \
                --num_intent_detection \
                --use_crf \
                --only_intent 0.1 \
                --max_freq 10

# Fine tune the whole model
python main.py  --model_type lstm \
                --model_dir misca \
                --task <mixatis or mixsnips> \
                --data_dir data \
                --do_train \
                --do_eval \
                --num_train_epochs 50 \
                --intent_loss_coef <lambda> \
                --learning_rate 1e-4 \
                --num_intent_detection \
                --use_crf \
                --max_freq 10 \
                --base_model dir_base \
                --intent_slot_attn_type coattention