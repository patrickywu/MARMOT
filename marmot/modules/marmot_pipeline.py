from marmot.modules.image_transformer import image_transformer
import torch
from torch import nn
from transformers import BertModel, BertConfig

class marmot(nn.Module):
    def __init__(self, bert_model, image_model, pretrained_image_channels=2048, pretrained_image_dim=7, bert_dim=768,
                dropout_p_final_clf=0.1, intermediate_layer_final_clf=768*4, num_classes=2):
        super().__init__()

        # Record the bert_model
        self.bert_model = bert_model

        # Image Translator
        self.ImageTranslator = image_transformer(self.bert_model, image_model, pretrained_image_channels=2048, pretrained_image_dim=7, bert_dim=768)

        # Encoder
        self.bert_clf = BertModel.from_pretrained(self.bert_model, output_attentions=True)
        triple_tti = nn.Embedding(3, 768)
        triple_tti.weight.data[:2].copy_(self.bert_clf.embeddings.token_type_embeddings.weight)
        triple_tti.weight.data[2].copy_(self.bert_clf.embeddings.token_type_embeddings.weight.data.mean(dim=0) +
                                        torch.randn(self.bert_clf.embeddings.token_type_embeddings.weight.data.mean(dim=0).size())*0.01)
        self.bert_clf.embeddings.token_type_embeddings = triple_tti

        # Classifier
        self.final_clf = nn.Sequential(nn.Linear(bert_dim, intermediate_layer_final_clf),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_p_final_clf),
                                       nn.Linear(intermediate_layer_final_clf, num_classes))

    def forward(self, img, pic, caption_ii, caption_tti, caption_am, text_ii, text_tti, text_am):

        # Translate image
        img_translated = self.ImageTranslator(img=img, caption_ii=caption_ii, caption_tti=caption_tti, caption_am=caption_am*pic.unsqueeze(-1))

        # Create position embeddings
        pos_text = torch.arange(text_ii.shape[1], dtype=torch.long).to(img.device)
        pos_text = pos_text.unsqueeze(0).expand(text_ii.shape[0], text_ii.shape[1])
        pos_caption = torch.arange(caption_ii.shape[1], dtype=torch.long).to(img.device)
        pos_caption = pos_caption.unsqueeze(0).expand(caption_ii.shape[0], caption_ii.shape[1])

        # Create token type IDs to input
        tti_caption = caption_tti + 1
        img_translated_tti = caption_tti + 2
        tti = torch.cat((text_tti, tti_caption, img_translated_tti), dim=1)

        # Create attention masks to input
        am = torch.cat((text_am, caption_am*pic.unsqueeze(-1), caption_am*pic.unsqueeze(-1)), dim=1)

        # Obtain inputs embeds
        text_embeds = self.bert_clf.embeddings.word_embeddings(text_ii)
        caption_embeds = self.bert_clf.embeddings.word_embeddings(caption_ii)
        img_translated_fts = img_translated.last_hidden_state + self.bert_clf.embeddings(input_ids=caption_ii, token_type_ids=caption_tti, position_ids=pos_caption)

        inputs_embeds = torch.cat((text_embeds, caption_embeds, img_translated_fts), dim=1)

        # Create position ids for the image
        pos_translated_img = torch.arange(img_translated.last_hidden_state.shape[1], dtype=torch.long).to(img.device)
        pos_translated_img = pos_translated_img.unsqueeze(0).expand(img_translated_fts.shape[0], img_translated_fts.shape[1])

        pos = torch.cat((pos_text, pos_caption, pos_translated_img), dim=1)

        # BERT output
        bert_output = self.bert_clf(attention_mask=am, token_type_ids=tti, position_ids=pos, inputs_embeds=inputs_embeds)

        # Classifier
        bert_output_avg = torch.sum(bert_output[0]*am.unsqueeze(-1), dim=1) / (torch.sum(am, dim=1).unsqueeze(-1))
        out = self.final_clf(bert_output_avg)

        return out, bert_output
