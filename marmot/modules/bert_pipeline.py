import torch
from torch import nn
from transformers import BertModel, BertConfig

class bert_text_only_pipeline(nn.Module):
    def __init__(self, bert_model, bert_dim=768, dropout_p_final_clf=0.1, intermediate_layer_final_clf=768*4, num_classes=2):
        super().__init__()
        # Record the bert_model
        self.bert_model = bert_model

        # Encoder
        self.bert_clf = BertModel.from_pretrained(self.bert_model, output_attentions=True)

        # Classifier
        self.final_clf = nn.Sequential(nn.Linear(bert_dim, intermediate_layer_final_clf),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_p_final_clf),
                                       nn.Linear(intermediate_layer_final_clf, num_classes))

    def forward(self, text_ii, text_tti, text_am):
        # BERT output
        bert_output = self.bert_clf(input_ids=text_ii, attention_mask=text_am, token_type_ids=text_tti)

        # Classifier
        bert_output_avg = torch.sum(bert_output[0]*text_am.unsqueeze(-1), dim=1) / (torch.sum(text_am, dim=1).unsqueeze(-1))
        out = self.final_clf(bert_output_avg)

        return out, bert_output

class bert_textcaption_pipeline(nn.Module):
    def __init__(self, bert_model, bert_dim=768, dropout_p_final_clf=0.1, intermediate_layer_final_clf=768*4, num_classes=2):
        super().__init__()
        # Record the bert_model
        self.bert_model = bert_model

        # Encoder
        self.bert_clf = BertModel.from_pretrained(self.bert_model, output_attentions=True)

        # Classifier
        self.final_clf = nn.Sequential(nn.Linear(bert_dim, intermediate_layer_final_clf),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_p_final_clf),
                                       nn.Linear(intermediate_layer_final_clf, num_classes))

    def forward(self, pic, caption_ii, caption_tti, caption_am, text_ii, text_tti, text_am):
        # Create position embeddings
        pos_text = torch.arange(text_ii.shape[1], dtype=torch.long).to(text_ii.device)
        pos_text = pos_text.unsqueeze(0).expand(text_ii.shape[0], text_ii.shape[1])
        pos_caption = torch.arange(caption_ii.shape[1], dtype=torch.long).to(text_ii.device)
        pos_caption = pos_caption.unsqueeze(0).expand(caption_ii.shape[0], caption_ii.shape[1])
        pos = torch.cat((pos_text, pos_caption), dim=1)

        # Create token type IDs to input
        tti_caption = caption_tti + 1
        tti = torch.cat((text_tti, tti_caption), dim=1)

        # Create attention masks to input
        am = torch.cat((text_am, caption_am*pic.unsqueeze(-1)), dim=1)

        # Obtain inputs embeds
        text_embeds = self.bert_clf.embeddings.word_embeddings(text_ii)
        caption_embeds = self.bert_clf.embeddings.word_embeddings(caption_ii)

        inputs_embeds = torch.cat((text_embeds, caption_embeds), dim=1)

        # BERT output
        bert_output = self.bert_clf(attention_mask=am, token_type_ids=tti, position_ids=pos, inputs_embeds=inputs_embeds)

        # Classifier
        bert_output_avg = torch.sum(bert_output[0]*am.unsqueeze(-1), dim=1) / (torch.sum(am, dim=1).unsqueeze(-1))
        out = self.final_clf(bert_output_avg)

        return out, bert_output
