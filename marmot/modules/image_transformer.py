import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class image_transformer(nn.Module):
    def __init__(self, bert_model, image_model, pretrained_image_channels=2048, pretrained_image_dim=7, bert_dim=768):
        """
        The image decoder transformer.
        """
        super().__init__()

        self.pretrained_image_model = image_model
        self.pretrained_image_channels = pretrained_image_channels
        self.pretrained_image_dim = pretrained_image_dim
        self.bert_dim = bert_dim
        self.encoder_img_feats = nn.Conv2d(pretrained_image_dim, bert_dim, 1)

        # Image Decoder
        ImageDecoder_config = BertConfig.from_pretrained(bert_model, is_decoder=True, add_cross_attention=True, output_attentions=True)
        self.ImageDecoder = BertModel.from_pretrained(bert_model, config=ImageDecoder_config)

    def forward(self, img, caption_ii, caption_tti, caption_am):

        # Process the image
        img_feats = self.pretrained_image_model(img)
        img_feats = img_feats.reshape(img.shape[0], self.pretrained_image_channels, self.pretrained_image_dim, self.pretrained_image_dim)
        img_feats_encoder = self.encoder_img_feats(img_feats)
        img_feats_encoder = img_feats_encoder.flatten(2).permute(0,2,1)

        # Feed caption through image decoder and take in image_encoder_features through the cross-attention mechanism
        image_decoder_features = self.ImageDecoder(input_ids=caption_ii, token_type_ids=caption_tti, attention_mask=caption_am, encoder_hidden_states=img_feats_encoder, return_dict=True)

        return image_decoder_features
