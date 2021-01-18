from transformers import BertTokenizer
from torch import nn

class text_processor(nn.Module):
    def __init__(self, bert_model, device):
        super().__init__()

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def extract_bert_inputs(self, text):
        encoded_text = self.tokenizer(text=text,
                                      add_special_tokens=True,
                                      padding=True,
                                      return_tensors='pt')

        ii_text = encoded_text['input_ids']
        tti_text = encoded_text['token_type_ids']
        am_text = encoded_text['attention_mask']

        ii_text = ii_text.to(self.device)
        tti_text = tti_text.to(self.device)
        am_text = am_text.to(self.device)

        return ii_text, tti_text, am_text
