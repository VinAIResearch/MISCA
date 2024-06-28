import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier, SlotClassifier, LSTMEncoder
from .attention import init_attention_layer, perform_attention, HierCoAttention

class JointRoberta(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst, slot_hier):
        super(JointRoberta, self).__init__(config)
        self.args = args
        self.attn_type = args.intent_slot_attn_type
        self.n_levels = args.n_levels
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.slot_hier = [len(x) for x in slot_hier]
        self.roberta = RobertaModel(config)
        self.lstm_intent = LSTMEncoder(
            config.hidden_size,
            args.decoder_hidden_dim,
            args.dropout_rate
        )
        self.lstm_slot = LSTMEncoder(
            config.hidden_size,
            args.decoder_hidden_dim,
            args.dropout_rate
        )

        self.intent_detection = IntentClassifier(self.num_intent_labels, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(
            args.decoder_hidden_dim,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.max_seq_len,
            self.args.slot_decoder_size,
            args.dropout_rate,
        )
        self.output_size = args.decoder_hidden_dim
        self.attention_mode = args.attention_mode
        
        if args.intent_slot_attn_type == 'coattention':
            dims = [self.args.label_embedding_size] + [args.slot_decoder_size] + [args.slot_decoder_size + args.level_projection_size] * (len(self.slot_hier) - 1) + [self.args.label_embedding_size]
            self.attn = HierCoAttention(args, dims, args.intent_slot_attn_size, args.dropout_rate)
        if args.intent_slot_attn_type:
            self.intent_refine = nn.Linear(args.decoder_hidden_dim + args.intent_slot_attn_size, self.num_intent_labels, args.dropout_rate)
            self.slot_refine = IntentClassifier(args.slot_decoder_size + args.intent_slot_attn_size, self.num_slot_labels, args.dropout_rate)
            self.slot_proj = IntentClassifier(self.num_slot_labels, self.args.label_embedding_size, args.dropout_rate)
            self.intent_proj = IntentClassifier(1, self.args.label_embedding_size, args.dropout_rate)

        init_attention_layer(self, 'intent', [self.num_intent_labels], 1, args.decoder_hidden_dim)
        if args.intent_slot_attn_type == 'coattention':
            init_attention_layer(self, 'slot', self.slot_hier, len(self.slot_hier), self.args.slot_decoder_size)

        self.relu = nn.LeakyReLU(0.2)
        self.intent_classifier = nn.Linear(args.decoder_hidden_dim, 1, args.dropout_rate)
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        mask = x.unsqueeze(0) < length.unsqueeze(1)
        # mask[:, 0] = 0
        return mask

    def forward(self, input_ids, attention_mask, token_type_ids, heads, intent_label_ids, slot_labels_ids, seq_lens):
        lens = torch.sum(attention_mask, dim=-1).cpu()
        encoded = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        intent_output = self.lstm_intent(encoded, lens)
        slot_output = self.lstm_slot(encoded, lens)

        intent_output = torch.cat(
            [torch.index_select(intent_output[i], 0, heads[i]).unsqueeze(0) for i in range(intent_output.size(0))],
            dim=0
        )
        slot_output = torch.cat(
            [torch.index_select(slot_output[i], 0, heads[i]).unsqueeze(0) for i in range(slot_output.size(0))],
            dim=0
        )

        i_context_vector, intent_logits, i_attn = perform_attention(self, 'intent', intent_output, None, [self.num_intent_labels], 1)
        intent_logits = intent_logits[-1]
        
        i_context_vector = i_context_vector[-1]
        intent_dec = self.intent_detection(intent_logits)
        x, slot_logits = self.slot_classifier(slot_output)

        if self.args.intent_slot_attn_type == 'coattention':
            s_context_vector, s_logits, s_attn = perform_attention(self, 'slot', x, None, self.slot_hier, len(self.slot_hier))

        if self.attn_type == 'coattention':
            if self.args.embedding_type == 'soft':
                slots = self.slot_proj(F.softmax(slot_logits, -1))
                intents = self.intent_proj(F.sigmoid(intent_logits.unsqueeze(2)))
            else:
                slot_label = torch.argmax(slot_logits, dim=-1)
                hard_label = F.one_hot(slot_label, num_classes=self.num_slot_labels)
                for i in range(len(seq_lens)):
                    hard_label[i, seq_lens[i]:, :] = 0
                slots = self.slot_proj(hard_label.float())
                
                int_labels = torch.zeros_like(intent_logits)
                num = torch.argmax(intent_dec, dim=-1)
                for i in range(len(intent_logits)):
                    num_i = num[i]
                    ids = torch.topk(intent_logits[i], num_i).indices
                    int_labels[i, ids] = 1.0
                
                intents = self.intent_proj(int_labels.unsqueeze(2))
            intent_vec, slot_vec = self.attn([intents] + s_context_vector + [slots])

        if self.attn_type:
            intent_logits = self.intent_refine.weight.mul(torch.tanh(torch.cat([i_context_vector, intent_vec], dim=-1))).sum(dim=2).add(self.intent_refine.bias)
            slot_logits = self.relu(self.slot_refine(torch.cat([x, self.relu(slot_vec)], dim=-1)))

        max_len = torch.max(seq_lens)
        attention_mask = self.sequence_mask(seq_lens, max_length=max_len)
        total_loss = 0
        aux_loss = 0
        intent_loss = 0
        slot_loss = 0
        count_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.BCEWithLogitsLoss()
                intent_loss_cnt = nn.CrossEntropyLoss()
                intent_count = torch.sum(intent_label_ids, dim=-1).long()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.float()) 
                count_loss = intent_loss_cnt(intent_dec.view(-1, self.num_intent_labels), intent_count)
            total_loss += intent_loss * self.args.intent_loss_coef 
            aux_loss += count_loss
        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte().to(slot_logits.device), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    # print("SHAPE", slot_labels_ids.shape, slot_logits.shape, active_loss.shape)
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.reshape(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += slot_loss * (1 - self.args.intent_loss_coef)
        total_loss += aux_loss * self.args.aux_loss_coef
        outputs = ((intent_logits, slot_logits, intent_dec),)  # add hidden states and attention if they are here
        outputs = ((total_loss, intent_loss, slot_loss, count_loss),) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
