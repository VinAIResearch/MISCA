import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Biaffine

class AttentionLayer(nn.Module):

    def __init__(self,
                 args,
                 size: int,
                 level_projection_size: int = 0,
                 n_labels=None,
                 n_level: int = 1
                 ):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(AttentionLayer, self).__init__()
        self.attention_mode = args.attention_mode

        self.size = size
        # For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
        # d_a is the output size of the first linear layer
        self.d_a = args.d_a if args.d_a > 0 else self.size

        # r is the number of attention heads

        self.n_labels = n_labels
        self.n_level = n_level

        self.level_projection_size = level_projection_size

        self.linear = nn.Linear(self.size, self.size, bias=False)
        
        self.first_linears = nn.ModuleList([nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)])
        self.second_linears = nn.ModuleList([nn.Linear(self.d_a, self.n_labels[label_lvl], bias=False) for label_lvl in range(self.n_level)])
        self.third_linears = nn.ModuleList([nn.Linear(self.size +
                                            (self.level_projection_size if label_lvl > 0 else 0),
                                            self.n_labels[label_lvl], bias=True) for label_lvl in range(self.n_level)])

        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal(first_linear.weight, mean, std)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        for linear in self.third_linears:
            torch.nn.init.normal(linear.weight, mean, std)

    def forward(self, x, previous_level_projection=None, label_level=0, masks=None):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]

        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        weights = F.tanh(self.first_linears[label_level](x))

        att_weights = self.second_linears[label_level](weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        context_vector = att_weights @ x

        batch_size = context_vector.size(0)

        if previous_level_projection is not None:
            temp = [context_vector,
                    previous_level_projection.repeat(1, self.n_labels[label_level]).view(batch_size, self.n_labels[label_level], -1)]
            context_vector = torch.cat(temp, dim=2)

        weighted_output = self.third_linears[label_level].weight.mul(context_vector).sum(dim=2).add(
            self.third_linears[label_level].bias)

        return context_vector, weighted_output, att_weights

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)

def init_attention_layer(model, name, n_labels, n_levels, output_size):
    
    model.level_projection_size = model.args.level_projection_size
    if model.attention_mode is not None:
        model.add_module(f'attention_{name}', AttentionLayer(args=model.args, size=output_size,
                                                            level_projection_size=model.level_projection_size,
                                                            n_labels=n_labels, n_level=n_levels))
    linears = []
    projection_linears = []
    for level in range(n_levels):
        level_projection_size = 0 if level == 0 else model.level_projection_size
        linears.append(nn.Linear(output_size + level_projection_size,
                                    n_labels[level]))
        projection_linears.append(nn.Linear(n_labels[level], model.level_projection_size, bias=False))
    model.add_module(f'linears_{name}', nn.ModuleList(linears))
    model.add_module(f'projection_linears_{name}', nn.ModuleList(projection_linears))
   


def perform_attention(model, name, all_output, last_output, n_labels, n_levels):
    attention_weights = None
    previous_level_projection = None
    weighted_outputs = []
    attention_weights = []
    context_vectors = []
    for level in range(n_levels):
        context_vector, weighted_output, attention_weight = model.__getattr__(f'attention_{name}')(all_output,
                                                            previous_level_projection, label_level=level)

        previous_level_projection = model.__getattr__(f'projection_linears_{name}')[level](
            torch.sigmoid(weighted_output) if model.attention_mode in ["label", "caml"]
            else torch.softmax(weighted_output, 1))
        previous_level_projection = F.sigmoid(previous_level_projection)
        weighted_outputs.append(weighted_output)
        attention_weights.append(attention_weight)
        context_vectors.append(context_vector)
        
    return context_vectors, weighted_outputs, attention_weights

class AttentionFlow(nn.Module):
    def __init__(self, args, dim_x, dim_y, out_dim, dropout_rate=0.):
        super(AttentionFlow, self).__init__()

        self.linear_x = nn.Linear(dim_x, out_dim)
        self.linear_y = nn.Linear(dim_y, out_dim)
        self.scorer = Biaffine(dim_y, dim_x, dropout=args.dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, y):
        # x : intent
        # y : seq_len
        # x = [bz, num_intent, out_dim]
        score = self.scorer(y, x)

        x = self.linear_x(x)
        x = self.dropout(x)
        # y = [bz, seq_len, out_dim]
        y = self.linear_y(y)
        y = self.dropout(y)

        # [bz, seq_len, num_intent]
        a = F.softmax(score, dim=-1)
        b = F.softmax(score.transpose(1, 2), dim=-1)

        out_slot = torch.tanh(torch.bmm(a, x))
        out_intent = torch.tanh(torch.bmm(b, y))

        return out_intent, out_slot

class HierCoAttention(nn.Module):
    def __init__(self, args, dims, out_dim, dropout_rate=0.):
        super(HierCoAttention, self).__init__()

        self.n_layers = len(dims)
        self.linears = nn.ModuleList([nn.Linear(inp_dim, out_dim, bias=True) for inp_dim in dims])
        self.reverse = nn.ModuleList([nn.Linear(inp_dim, out_dim, bias=True) for inp_dim in dims])

        self.scorers = nn.ModuleList([Biaffine(dims[i], dims[i + 1], dropout=dropout_rate) for i in range(self.n_layers - 1)])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inps):
        # inps should be list of [intent, ..., slots]
        assert len(inps) == self.n_layers
        Cs = []
        for i in range(self.n_layers - 1):
            Cs.append(self.scorers[i](inps[i], inps[i + 1]))
        
        projs = []
        revers = []
        for i in range(self.n_layers):
            projs.append(self.linears[i](inps[i]))
            revers.append(self.reverse[i](inps[i]))
        
        slots = None
        for i in range(self.n_layers - 1):
            if slots is None:
                slots = torch.tanh(torch.bmm(Cs[0].transpose(1, 2), projs[0]) + projs[1])
            else:
                slots = torch.bmm(Cs[i].transpose(1, 2), slots) + projs[i + 1]
                if i < self.n_layers - 2:
                    slots = torch.tanh(slots)
        # slots = self.dropout(slots)
        
        intents = None
        for i in range(self.n_layers - 1, 0, -1):
            if intents is None:
                intents = torch.tanh(torch.bmm(Cs[-1], revers[-1]) + revers[-2])
            else:
                intents = torch.bmm(Cs[i - 1], intents) + revers[i - 1]
                if i > 1:
                    intents = torch.tanh(intents)
        return intents, slots