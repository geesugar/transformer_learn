from transformers import BertModel, BertTokenizer
import torch



model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


length = 20

sentence = "I love Paris"
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
tokens = tokens + ['[PAD]' for _ in range(length - len(tokens))]
attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

last_hidden_state, pooler_output, hidden_states = model(token_ids, attention_mask=attention_mask)

print(token_ids)
print(attention_mask)