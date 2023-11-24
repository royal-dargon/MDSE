"""
codes about the model 2.0
writen by Jingzhe Li 2023.4
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForPreTraining, RobertaModel, RobertaConfig
import torchvision.models as cv_models
# from fairseq.model.robeata import RobertaModel

# from util.compute_score import similarity
# from pre_model import RobertaEncoder



class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token




class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = ''

        if opt.text_model == 'bert-en':
            self.config = BertConfig.from_pretrained(abl_path + 'bert_en/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert_en/', config=self.config)
            self.model = self.model.bert
        elif opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert_base/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert_base/', config=self.config)
            self.model = self.model.bert
        elif opt.text_model == 'roberta':
            self.config = RobertaConfig.from_pretrained('./bonemodel/roberta_sa/roberta-base/')
            self.model = RobertaModel.from_pretrained('./bonemodel/roberta_sa/roberta-base/', config=self.config)

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = 768

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, token_id, attention_mask):
        output = self.model(token_id, attention_mask=attention_mask)
        # text_encoder = self.model.extract_features(token_id)
        """
        text_encoder:(batch_size, length, 768)
        text_cls:(batch_size, 768)
        new_global(batch, 768)
        """
        text_encoder = output.last_hidden_state
        return text_encoder


class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        self.vit = cv_models.vit_b_16(pretrained=False)
        self.vit.load_state_dict(torch.load('bonemodel/vit_b_16.pth'))
        self.output_dim = 768

        for param in self.vit.parameters():
            if opt.fixed_image_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        feats = self.vit._process_input(images)
        batch_class_token = self.vit.class_token.expand(images.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)
        feats = self.vit.encoder(feats)
        """
        image_encoder:(batch_size, 49, 2048)
        image_cls:(batch_size, 2048)
        new_global:(batch_size, 2048)
        """
        return feats


"""
目前确定的backbone是文本使用BERT，图像使用ResNet进行特征提取
"""


class CrossAttention(nn.Module):
    def __init__(self, opt):
        super(CrossAttention, self).__init__()
        self.multi_attn = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.linear1 = nn.Linear(opt.tran_dim, opt.tran_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(opt.tran_dim, opt.tran_dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(opt.tran_dim)
        self.norm2 = nn.LayerNorm(opt.tran_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, k_v, q):
        k_v = self.norm1(k_v)
        q = self.norm1(q)
        attn_out, _ = self.multi_attn(q, k_v, k_v)
        attn_out = self.dropout(attn_out)
        attn_out = q + attn_out
        x = attn_out
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        attn_out = attn_out + x
        return attn_out


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type

        """
        使用Roberta进行实验
        """
        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)
        self.text_image_encoder = CrossAttention(opt)
        self.image_text_encoder = CrossAttention(opt)
        self.text_single_encoder = CrossAttention(opt)
        self.image_single_encoder = CrossAttention(opt)
        self.text_image_decoder = CrossAttention(opt)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        # self.text_image_decoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask):
        text_init = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        image_init = self.image_model(image_inputs)

        text_guide_image = self.text_image_encoder(text_init, image_init)
        image_guide_text = self.image_text_encoder(image_init, text_init)

        # 这里是对除去cls进行相似度计算
        text_guide_image_temp = text_guide_image[:, 1:, :].permute(0, 2, 1).contiguous()
        image_guide_text_temp = image_guide_text[:, 1:, :].permute(0, 2, 1).contiguous()


        """
        计算相似度：text_single = (batch_size, text_length, 768) * (batch_size, 768, text_length)
        eg.(64, 64, 64)
        计算相似度：image_single = (batch_size, image_length, 768) * (batch_size, 768, text_length)
        eg.(64, 196, 196)
        """
        text_single_mask = torch.matmul(text_init[:, 1:, :], text_guide_image_temp)
        image_single_mask = torch.matmul(image_init[:, 1:, :], image_guide_text_temp)

        text_single_mask = torch.sum(text_single_mask, dim=-1)
        image_single_mask = torch.sum(image_single_mask, dim=-1)
        text_single_mask = 1 - torch.softmax(text_single_mask, dim=-1)
        image_single_mask = 1 - torch.softmax(image_single_mask, dim=-1)

        text_single = torch.mul(text_init[:, 1:, :], text_single_mask.unsqueeze(dim=-1))
        image_single = torch.mul(image_init[:, 1:, :], image_single_mask.unsqueeze(dim=-1))

        text_single = self.text_single_encoder(text_single, text_single)
        image_single = self.text_single_encoder(image_single, image_single)

        """
        text_guide_image: (batch_size, word_length + 1, 768) 
        image_guide_text: (batch_size, 196 + 1, 768)
        text_single: (batch_size, word_length, 768)
        image_single:(batch_size, 196, 768)
        """
        text_single_output = torch.mean(text_single, dim=1)
        image_single_output = torch.mean(image_single, dim=1)

        text_image = torch.cat((image_guide_text, text_guide_image), dim=1)
        text_image = self.text_image_decoder(text_image, text_image)

        text_image_output = torch.mean(text_image, dim=1)

        return text_image_output, text_single_output, image_single_output




class Classification(nn.Module):
    def __init__(self, opt):
        super(Classification, self).__init__()
        self.fuse_model = FuseModel(opt)
        self.temperature = opt.temperature
        self.set_cuda = opt.cuda

        self.origin_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim * 2, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.text_image_linear = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim * 2, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.image_text_linear = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim * 2, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim * 2, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

        self.momentum = 0.995

        self.model_pairs = []  # 初始化 model_pairs 属性
        # 创建模型对并将当前模型的参数复制到存储模型中
        # for _ in range(num_model_pairs):  # 根据需要设置模型对的数量
        model = copy.deepcopy(self.fuse_model)
        model_m = copy.deepcopy(self.fuse_model)
        model_m.load_state_dict(model.state_dict())  # 复制当前模型的参数到存储模型
        self.model_pairs.append((model, model_m))  # 将模型对添加到列表中

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    def forward(self, data_origin, labels=None, target_labels=None, kind=None):

        origin_res, origin_text_cls, origin_image_cls = self.fuse_model(data_origin.texts,
                                                                        data_origin.bert_attention_mask,
                                                                        data_origin.images,
                                                                        data_origin.text_image_mask)
        text_image_single = self.text_image_linear(torch.cat((origin_text_cls, origin_res), dim=-1))
        # text_image_single = self.text_image_linear(torch.add(origin_text_cls, origin_res))
        image_text_single = self.image_text_linear(torch.cat((origin_res, origin_image_cls), dim=-1))
        # image_text_single = self.image_text_linear(torch.add(origin_res, origin_image_cls))
        origin_res = torch.cat((text_image_single, image_text_single), dim=-1)
        # origin_res = torch.add(text_image_single, image_text_single)
        output = self.output_classify(origin_res)

        if kind == 'kkk':
            """
            对比学习，基于特征实现
            第一步是将多模态特征，文本特征，图像特征进行拼接，获得(batch_size * 3, 768)
            第二步使用torch.einsum，计算相似性得分，获得(batch * 3, batch * 3)大小的矩阵
            第三步使用softmax处理
            第三步使用view(-1)，使大小变为(batch * 3 * batch * 3)
            接下来是获取对每个样本寻找他们的正类
            """
            # multi_res = self.origin_linear_change(origin_res)
            # print(multi_res.shape, text_image_single.shape)
            # all_res = torch.cat((multi_res, text_image_single), dim=0)
            # all_res = torch.cat((multi_res, image_text_single), dim=0)
            all_res = torch.cat((text_image_single, image_text_single), dim=0)
            l_pos_neg = torch.einsum('nc,ck->nk', [all_res, all_res.T])
            l_pos_neg_self = torch.log_softmax(l_pos_neg, dim=-1)
            l_pos_neg_self = l_pos_neg_self.view(-1)
            cl_self_labels = target_labels[labels[0]]
            cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[0]] + labels.size(0)), dim=0)
            cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[0]] + labels.size(0) * 2), dim=0)
            for index in range(1, all_res.size(0)):
                i = index % origin_res.size(0)
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[i]] + index * origin_res.size(0)), dim=0)
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[i]] + index * origin_res.size(0) +
                                             labels.size(0)), dim=0)
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[i]] + index * origin_res.size(0) +
                labels.size(0) * 2), dim=0)
            l_pos_neg_self = l_pos_neg_self / self.temperature
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)



            return output, cl_self_loss
        return output









