import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import clip
from einops import rearrange, repeat
import kornia
import torchvision

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

class ImageEncoderVGG16BN(torch.nn.Module):
    # "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    # conv1_1: conv, bn, relu 3
    # conv1_2: conv, bn, relu 6
    # "M": maxpool 7
    # conv2_1: conv, bn, relu 10
    # conv2_2: conv, bn, relu 13
    # "M": maxpool 14
    # conv3_1: conv, bn, relu 17
    # conv3_2: conv, bn, relu 20   
    # conv3_3: conv, bn, relu 23
    # "M": maxpool 24
    # conv4_1: conv, bn, relu 27
    # conv4_2: conv, bn, relu 30   
    # conv4_3: conv, bn, relu 33
    # "M": maxpool 34
    # conv5_1: conv, bn, relu 37
    # conv5_2: conv, bn, relu 40   
    # conv5_3: conv, bn, relu 43
    # "M": maxpool 44

    def __init__(self, requires_grad=True):
        super(ImageEncoderVGG16BN, self).__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True)

        vgg_features = vgg.features
        self.conv1_2 = vgg_features[:4]
        self.conv2_2 = vgg_features[4:11]
        self.conv3_3 = vgg_features[11:21]
        self.conv4_3 = vgg_features[21:31]
        self.conv5_3 = vgg_features[31:41]
        self.conv_last = vgg_features[41:44]
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 4 * 4, 128)
        self.trans1_2 = nn.Conv2d(64, 192, 1)
        self.trans2_2 = nn.Conv2d(128, 384, 1)
        self.trans3_3 = nn.Conv2d(256, 384, 1)
        self.trans4_3 = nn.Conv2d(512, 768, 1)
        self.trans5_3 = nn.Conv2d(512, 768, 1)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, img):
        
        img = (img + 1) / 2. # [-1, 1] -> [0, 1]
        img = (img - self.mean) / self.std
        
        # conv_feats = []
        conv1_2 = self.conv1_2(img)
        conv2_2 = self.conv2_2(conv1_2)
        conv3_3 = self.conv3_3(conv2_2)
        conv4_3 = self.conv4_3(conv3_3)
        conv5_3 = self.conv5_3(conv4_3)
        conv_last = self.conv_last(conv5_3)
        output = {}

        output['f1'] = F.interpolate(self.trans1_2(conv1_2), size=[16, 16]).repeat(1, 1, 4, 4)
        output['f2'] = F.interpolate(self.trans2_2(conv2_2), size=[8, 8]).repeat(1, 1, 4, 4)
        output['f3'] = F.interpolate(self.trans3_3(conv3_3), size=[4, 4]).repeat(1, 1, 4, 4)
        output['f4'] = F.interpolate(self.trans4_3(conv4_3), size=[2, 2]).repeat(1, 1, 4, 4)
        output['f5'] = F.interpolate(self.trans5_3(conv5_3), size=[1, 1]).repeat(1, 1, 4, 4)


        return output