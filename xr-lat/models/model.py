import torch
from torch.nn import BCEWithLogitsLoss, Dropout, Linear
from transformers import AutoModel, XLNetModel
from .losses import AsymmetricLoss

class LableWiseAttentionLayer(torch.nn.Module):
    def __init__(self, model_args):
        super(LableWiseAttentionLayer, self).__init__()

        self.model_args = model_args

        # layers
        self.l1_linear = torch.nn.Linear(self.model_args.d_model,
                                         self.model_args.d_model, bias=False)
        self.tanh = torch.nn.Tanh()
        self.l2_linear = torch.nn.Linear(self.model_args.d_model, self.model_args.num_labels, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

        self.hyperbolic_linear = torch.nn.Linear(self.model_args.hyperbolic_dim, self.model_args.d_model, bias=False)
        self.label_query_vectors = None

        # Mean pooling last hidden state of code title from transformer model as the initial code vectors
        self._init_linear_weights(mean=self.model_args.linear_init_mean, std=self.model_args.linear_init_std)


    def _init_linear_weights(self, mean, std):
        # normalize the l1 weights
        torch.nn.init.normal_(self.l1_linear.weight, mean, std)
        if self.l1_linear.bias is not None:
            self.l1_linear.bias.data.fill_(0)
        # initialize the l2
        torch.nn.init.normal_(self.l2_linear.weight, mean, std)
        if self.l2_linear.bias is not None:
            self.l2_linear.bias.data.fill_(0)

        torch.nn.init.normal_(self.hyperbolic_linear.weight, mean, std)
        if self.hyperbolic_linear.bias is not None:
            self.hyperbolic_linear.bias.data.fill_(0)


    def forward(self, x):
        # input: (batch_size, max_seq_length, transformer_hidden_size)
        # output: (batch_size, max_seq_length, transformer_hidden_size)
        # Z = Tan(WH)
        l1_output = self.tanh(self.l1_linear(x))

        # when use hyperbolic, recalculate the query vectors in l2_linear
        if self.model_args.bootstrapping and self.model_args.use_hyperbolic:
            if self.label_query_vectors is None:
                attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
            else:
                label_query = self.hyperbolic_linear(self.label_query_vectors)
                label_query = self.l2_linear.weight + label_query
                attention_weight = self.softmax(torch.matmul(l1_output, label_query.transpose(0, 1))).transpose(1, 2)
        else:
            # softmax(UZ)
            # l2_linear output shape: (batch_size, max_seq_length, num_labels)
            if self.label_indices is None:
                # attention_weight shape: (batch_size, num_labels, max_seq_length)
                attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
            else:
                # num_labels will be the active labels (including padding indices)
                l2_W = self.l2_linear.weight  # (num_labels, hidden_size)
                l2_W_zeros = torch.zeros((1, l2_W.shape[1]), requires_grad=False)
                l2_W_zeros = l2_W_zeros.to(l2_W.device)
                l2_W = torch.vstack((l2_W, l2_W_zeros))
                # if self.model_args.chunk_attention:
                #     # expand the label_indices according to num_chunks
                #     dim_0 = self.label_indices.size(dim=0)
                #     dim_1 = self.label_indices.size(dim=1)
                #     self.label_indices = self.label_indices.view(dim_0, 1, dim_1)
                #     self.label_indices = self.label_indices.expand(dim_0, self.model_args.num_chunks_per_document, dim_1)
                #     self.label_indices = self.label_indices.reshape(dim_0*self.model_args.num_chunks_per_document, -1)

                # get subset by label_indices
                l2_W = l2_W[self.label_indices]
                # shape: (batch_size, max_seq_length, num_act_labels)
                attention_weight = torch.matmul(l1_output, torch.transpose(l2_W, 1, 2))
                # output shape: (batch_size, num_act_labels, max_seq_length)
                attention_weight = self.softmax(attention_weight).transpose(1, 2)


        # attention_output shpae: (batch_size, num_labels/num_act_labels, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight


class ChunkAttentionLayer(torch.nn.Module):
    def __init__(self, model_args):
        super(ChunkAttentionLayer, self).__init__()

        self.model_args = model_args

        # layers
        self.l1_linear = torch.nn.Linear(self.model_args.d_model,
                                         self.model_args.d_model, bias=False)
        self.tanh = torch.nn.Tanh()
        self.l2_linear = torch.nn.Linear(self.model_args.d_model, 1, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

        self._init_linear_weights(mean=self.model_args.linear_init_mean, std=self.model_args.linear_init_std)

    def _init_linear_weights(self, mean, std):
        # initialize the l1
        torch.nn.init.normal_(self.l1_linear.weight, mean, std)
        if self.l1_linear.bias is not None:
            self.l1_linear.bias.data.fill_(0)
        # initialize the l2
        torch.nn.init.normal_(self.l2_linear.weight, mean, std)
        if self.l2_linear.bias is not None:
            self.l2_linear.bias.data.fill_(0)

    def forward(self, x):
        # input: (batch_size, num_chunks, transformer_hidden_size)
        # output: (batch_size, num_chunks, transformer_hidden_size)
        # Z = Tan(WH)
        l1_output = self.tanh(self.l1_linear(x))
        # softmax(UZ)
        # l2_linear output shape: (batch_size, num_chunks, 1)
        # attention_weight shape: (batch_size, 1, num_chunks)
        attention_weight = self.softmax(self.l2_linear(l1_output)).transpose(1, 2)
        # attention_output shpae: (batch_size, 1, transformer_hidden_size)
        attention_output = torch.matmul(attention_weight, x)

        return attention_output, attention_weight

# define the model class
class HiLATModel(torch.nn.Module):
    def __init__(self, model_args):
        super(HiLATModel, self).__init__()

        self.model_args = model_args
        # layers
        self.transformer_layer = AutoModel.from_pretrained(self.model_args.transformer_name_or_path)
        if isinstance(self.transformer_layer, XLNetModel):
            self.transformer_layer.config.use_mems_eval = False
        self.dropout = Dropout(p=self.model_args.dropout)

        self.label_wise_attention_layer = LableWiseAttentionLayer(self.model_args)

        self.dropout_att = Dropout(p=self.model_args.dropout_att)

        # initial chunk attention
        if self.model_args.chunk_attention:
            self.chunk_attention_layer = ChunkAttentionLayer(self.model_args)

        self.classifier_layer = Linear(self.model_args.d_model,
                                       self.model_args.num_labels)

        self.sigmoid = torch.nn.Sigmoid()

        if self.model_args.transformer_layer_update_strategy == "no":
            self.freeze_all_transformer_layers()
        elif self.model_args.transformer_layer_update_strategy == "last":
            self.freeze_all_transformer_layers()
            self.unfreeze_transformer_last_layers()

        # initialize the weights of classifier
        self._init_linear_weights(mean=self.model_args.linear_init_mean, std=self.model_args.linear_init_std)

        # if use Asymmetric loss function
        if self.model_args.use_ASL:
            self.loss_fct = AsymmetricLoss()
            print("ASL loss")
        else:
            self.loss_fct = BCEWithLogitsLoss()
            print("BCE Loss")


    def _init_linear_weights(self, mean, std):
        torch.nn.init.normal_(self.classifier_layer.weight, mean, std)

    def init_transformer_from_model(self, parent_model):
        self.transformer_layer = parent_model.transformer_layer

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, targets=None, label_indices=None, label_values=None):
        # input ids/mask/type_ids shape: (batch_size, num_chunks, max_seq_length)
        batch_size, num_chunks, max_seq_length = input_ids.size()
        hidden_size = self.model_args.d_model
        # output: (batch_size*num_chunks, max_seq_length, hidden_size)
        l1_output = self.transformer_layer(input_ids=input_ids.view(-1, max_seq_length),
                                           attention_mask=attention_mask.view(-1, max_seq_length),
                                           token_type_ids=token_type_ids.view(-1, max_seq_length))

        # dropout transformer output
        l2_dropout = self.dropout(l1_output[0])
        # Label-wise attention layers. output: (batch_size, num_chunks, num_labels, hidden_size)

        # subset labels
        self.label_wise_attention_layer.label_indices = label_indices
        if self.model_args.chunk_attention:
            # # l3_attention output: (batch_size*num_chunks, num_labels, hidden_size)
            # # l3_attention_weight output: (batch_size*num_chunks, num_labels, max_seq_length)
            # l3_attention, l3_attention_weight = self.label_wise_attention_layer(l2_dropout)
            # l3_dropout = self.dropout_att(l3_attention)
            # # Chunk attention layers
            # # output: (batch_size, num_labels/num_act_labels, hidden_size)
            # l4_chunk_attention, l4_chunk_attention_weights = self.chunk_attention_layer(
            #     l3_dropout.view(-1, num_chunks, hidden_size))
            # # output shape: (batch_size, num_labels, hidden_size)
            # l4_dropout = self.dropout_att(l4_chunk_attention)
            # l4_dropout = l4_dropout.view(batch_size, -1, hidden_size)

            l2_dropout = l2_dropout.reshape(batch_size, num_chunks, max_seq_length, -1)
            # Label-wise attention layers
            # output: (batch_size, num_chunks, num_labels, hidden_size)
            attention_output = []
            attention_weights = []

            for i in range(num_chunks):
                # input: (batch_size, max_seq_length, transformer_hidden_size)
                attention_layer = self.label_wise_attention_layer
                l3_attention, attention_weight = attention_layer(l2_dropout[:, i, :])
                # l3_attention shape: (batch_size, num_labels, hidden_size)
                # attention_weight: (batch_size, num_labels, max_seq_length)
                attention_output.append(l3_attention)
                attention_weights.append(attention_weight)

            attention_output = torch.stack(attention_output)
            attention_output = attention_output.transpose(0, 1)
            attention_weights = torch.stack(attention_weights)
            attention_weights = attention_weights.transpose(0, 1)

            l3_dropout = self.dropout_att(attention_output)

            # chunk attention
            chunk_attention_output = []
            chunk_attention_weights = []
            for i in range(attention_output.size(dim=2)):
                chunk_attention = self.chunk_attention_layer
                l4_chunk_attention, l4_chunk_attention_weights = chunk_attention(l3_dropout[:, :, i])
                chunk_attention_output.append(l4_chunk_attention.squeeze(dim=1))
                chunk_attention_weights.append(l4_chunk_attention_weights.squeeze(dim=1))

            chunk_attention_output = torch.stack(chunk_attention_output)
            chunk_attention_output = chunk_attention_output.transpose(0, 1)
            chunk_attention_weights = torch.stack(chunk_attention_weights)
            chunk_attention_weights = chunk_attention_weights.transpose(0, 1)
            # output shape: (batch_size, num_labels, hidden_size)
            l4_dropout = self.dropout_att(chunk_attention_output)
        else:
            # l3_attention output: (batch_size, num_labels, hidden_size)
            # l3_attention_weight output: (batch_size, num_labels, max_seq_length)
            l3_attention, l3_attention_weight = self.label_wise_attention_layer(l2_dropout.view(batch_size, num_chunks*max_seq_length, -1))
            l4_dropout = self.dropout_att(l3_attention)

        # loss_fct = BCEWithLogitsLoss()

        if label_indices is None:
            logits = self.classifier_layer.weight.mul(l4_dropout).sum(dim=2).add(self.classifier_layer.bias)
            loss = self.loss_fct(logits, targets)
        else:
            # only calculate the active labels' logits
            classifier_W = self.classifier_layer.weight  # (num_labels, hidden_size)
            classifier_W_zeros = torch.zeros((1, classifier_W.shape[1]), requires_grad=False)
            classifier_W_zeros = classifier_W_zeros.to(classifier_W.device)
            classifier_W = torch.vstack((classifier_W, classifier_W_zeros))
            # get subset by label_indices
            classifier_W = classifier_W[label_indices] # (num_act_labels, hidden_size)
            # prepare bias
            classifier_bias = self.classifier_layer.bias
            classifier_bias_zeros = torch.zeros((1,), requires_grad=False)
            classifier_bias_zeros = classifier_bias_zeros.to(classifier_bias.device)
            classifier_bias = torch.hstack((classifier_bias, classifier_bias_zeros))
            classifier_bias = classifier_bias[label_indices]

            logits = classifier_W.mul(l4_dropout).sum(dim=2).add(classifier_bias)
            loss = self.loss_fct(logits, label_values)

        return {
            "loss": loss,
            "logits": logits,
            # "label_attention_weights": attention_weights,
            # "chunk_attention_weights": chunk_attention_weights if self.model_args.chunk_attention else []
            "label_indices": label_indices if label_indices is not None else []
        }

    def freeze_all_transformer_layers(self):
        """
        Freeze all layer weight parameters. They will not be updated during training.
        """
        for param in self.transformer_layer.parameters():
            param.requires_grad = False

    def unfreeze_all_transformer_layers(self):
        """
        Unfreeze all layers weight parameters. They will be updated during training.
        """
        for param in self.transformer_layer.parameters():
            param.requires_grad = True

    def unfreeze_transformer_last_layers(self):
        for name, param in self.transformer_layer.named_parameters():
            if "layer.11" in name or "pooler" in name:
                param.requires_grad = True
