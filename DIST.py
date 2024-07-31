import math
import copy
import torch
from torch import nn
from transformers import BertModel, AutoModel
import json

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}
AlbertLayerNorm = torch.nn.LayerNorm
BertLayerNorm = torch.nn.LayerNorm

class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num=1):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self,
                s1_hidden_states,
                s2_hidden_states,
                s2_attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states,
                                            s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states,
                                          s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor,
                                    s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class MultimodalityFusionLayer(nn.Module):
    def __init__(self, config, layernum=1):
        super(MultimodalityFusionLayer, self).__init__()
        block = MultimodalityFusionBlock(config)

        self.mfl = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(layernum)])

    def forward(self, query_hidden_state, kv_hidden_state,
                query_attention_mask, kv_attention_mask):
        all_encoder_layers = []
        for block in self.mfl:
            query_hidden_state = block(query_hidden_state, kv_hidden_state,
                                       query_attention_mask, kv_attention_mask)
            all_encoder_layers.append(query_hidden_state)
        return all_encoder_layers[-1]

class MultimodalityFusionBlock(nn.Module):
    def __init__(self, config):
        super(MultimodalityFusionBlock, self).__init__()
        self.self_attention = BertAttention(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.cross_attention = BertCoAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = DecoderOutput(config)

    # def forward(self, query_hidden_state, kv_hidden_state,
    #             query_attention_mask, kv_attention_mask):
    #     self_attention_output = self.self_attention(query_hidden_state,
    #                                                 query_attention_mask)
    #     self_attention_output = self.LayerNorm(self_attention_output +
    #                                            query_hidden_state)
    #     cross_modal_output = self.cross_attention(query_hidden_state,
    #                                               kv_hidden_state,
    #                                               kv_attention_mask)
    #     intermediate_output = self.intermediate(cross_modal_output)
    #     output = self.output(intermediate_output, self_attention_output)
    #
    #     return output

    def forward(self, query_hidden_state, kv_hidden_state,
                attention_mask, target_cls, extend_mask):
        #query_hidden_state = torch.cat((target_cls,query_hidden_state),dim=1)# 把需要判断的方面放在开头强化暗示

        self_attention_output = self.self_attention(query_hidden_state,
                                                    attention_mask)
        self_attention_output = self.LayerNorm(self_attention_output +
                                               query_hidden_state)
        cross_modal_output = self.cross_attention(query_hidden_state,
                                                  kv_hidden_state,
                                                  attention_mask)#标准长度的mask
        intermediate_output = self.intermediate(cross_modal_output)
        output = self.output(intermediate_output, self_attention_output)

        return output

class DecoderOutput(nn.Module):
    def __init__(self, config):
        super(DecoderOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DIST(nn.Module):
    def __init__(self,
                 config,
                 num_labels=8,
                 embedding=None,
                 pretrain='./pretrains',
                 config_cat='./pretrains/bert_config_cat.json'
                 ):
        super(DIST, self).__init__()
        self.num_labels = num_labels
        self.embedding = embedding
        self.sentence_extract = AutoModel.from_pretrained(pretrain)
        self.target_extract = AutoModel.from_pretrained(pretrain)
        self.left_cross = BertCrossEncoder(config, layer_num=2)
        self.right_cross = BertCrossEncoder(config, layer_num=2)
        self.oriented_left = MultimodalityFusionBlock(config)
        self.oriented_right = MultimodalityFusionBlock(config)
        self.img2bert = nn.Linear(1536,config.hidden_size)
        self.attention_cls = MultimodalEncoder(config_cat)#config_cat
        self.pooler = BertPooler(config_cat)#config_cat
        self.classifier = nn.Linear(config_cat.hidden_size,8)

    def forward(self, sentence_ids, sentence_mask, sentence_token_type_ids,
                target_ids, target_mask, target_token_type_ids, img_feature,
                img_mask):
        # 以下注释的代码主要是调整句子长度为128 用以完成参数设置实验
        # sentence_mask = sentence_mask[:, :128]
        # sentence_token_type_ids = sentence_token_type_ids[:, :128]
        #
        # sentence_ids_clipped = sentence_ids[:, :128].clone()
        # sentence_ids_clipped[:, 127] = torch.where(
        #     sentence_ids[:, 127] != 0,
        #     torch.tensor(2, dtype=sentence_ids.dtype, device=sentence_ids.device),
        #     sentence_ids_clipped[:, 127]
        # )
        # sentence_ids=sentence_ids_clipped

        sentence_output = self.sentence_extract(sentence_ids, attention_mask=sentence_mask, token_type_ids=sentence_token_type_ids)
        target_output = self.target_extract(target_ids, attention_mask=target_mask,token_type_ids=target_token_type_ids)
        #img_output=self.img_extract(inputs_embeds=img_feature,attention_mask=img_mask)
        sentence_feature = sentence_output.last_hidden_state
        target_feature = target_output.last_hidden_state
        img_feature = self.img2bert(img_feature)
        target_feature_cls = target_feature[:,0:1,:]

        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        extended_sentence_mask = sentence_mask.unsqueeze(1).unsqueeze(2)
        extended_sentence_mask = extended_sentence_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_sentence_mask = (1.0 - extended_sentence_mask) * -10000.0

        elements_to_add = torch.ones((target_mask.size(0), 1), dtype=target_mask.dtype,device=target_mask.device)
        # 在批次的每个前连接 1 标记target cls token
        target_oriented_mask = torch.cat((elements_to_add, target_mask), dim=1)
        output_mask = torch.cat((elements_to_add, target_oriented_mask), dim=1)

        extended_target_mask = target_mask.unsqueeze(1).unsqueeze(2)
        extended_target_mask = extended_target_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_target_mask = (1.0 - extended_target_mask) * -10000.0

        #下面这个mask是以情感为导向的拼接后的结果 长度相较于target的长度+1
        extended_target_oriented_mask = target_oriented_mask.unsqueeze(1).unsqueeze(2)
        extended_target_oriented_mask = extended_target_oriented_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_target_oriented_mask = (1.0 - extended_target_oriented_mask) * -10000.0

        extended_output_mask = output_mask.unsqueeze(1).unsqueeze(2)
        extended_output_mask = extended_output_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_output_mask = (1.0 - extended_output_mask) * -10000.0

        target_sentence = self.left_cross(target_feature, sentence_feature, extended_sentence_mask)[-1]# (b,6,768)
        target_image = self.right_cross(target_feature, img_feature, extended_img_mask)[-1]# (b,6,768)

        target_oriented_img = self.oriented_left(target_sentence,target_image,extended_target_mask,target_feature_cls,extended_target_oriented_mask)
        target_oriented_sentence = self.oriented_right(target_image,target_sentence,extended_target_mask,target_feature_cls,extended_target_oriented_mask)

        #arget_oriented_img_cls = target_oriented_img[:,0:1,:]

        target_oriented_feature = torch.cat((target_oriented_img, target_oriented_sentence),dim=-1)
        target_oriented_output = self.attention_cls(target_oriented_feature,extended_target_mask)[-1]

        output = self.pooler(target_oriented_output)
        logits = self.classifier(output)

        return logits