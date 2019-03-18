import torch
import spacy
from pprint import pprint
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
nlp = spacy.load('en_core_web_lg')
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text_without_separators = """Kathleen Nott was born in Camberwell, London. Her father, Philip, was a lithographic printer, and her mother, Ellen, ran a boarding house in Brixton; Kathleen was their third daughter. She was educated at Mary Datchelor Girls' School (now closed), London, before attending King's College, London."""
text = """[CLS] Kathleen Nott was born in Camberwell, London. [SEP] Her father, Philip, was a lithographic printer, and her mother, Ellen, ran a boarding house in Brixton; Kathleen was their third daughter. [SEP] She was educated at Mary Datchelor Girls' School (now closed), London, before attending King's College, London. [SEP]"""
tokenized_text = tokenizer.tokenize(text)
processed = nlp(text_without_separators)
pprint([ent.text for ent in processed.ents])
pprint([tokenizer.tokenize(ent.text) for ent in processed.ents])
pprint(tokenized_text)
pronoun_token_no = 48
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

tokenized_ents = [tokenizer.tokenize(ent.text) for ent in processed.ents]
n_layers = 12
n_heads = 12
index_of_pronoun = 48
pprint([ent.text for ent in processed.ents])
output = []
for l in range(n_layers):
    attn_vector = []
    for i, ent in enumerate(processed.ents):
        ent = ent.text
        tokenized_ent = tokenized_ents[i]
        subtokens_in_ent = []
        for subtoken in tokenized_ent:
            indexes_to_sum = [i for i, t in enumerate(tokenized_text) if t == subtoken]
            k = torch.squeeze(model.encoder.layer[l].attention.self.attention_scores)
            s = torch.squeeze(torch.sum(k.narrow(1, index_of_pronoun, 1), dim=0))
            subtoken_prob = torch.sum(s[indexes_to_sum])
            subtokens_in_ent.append(subtoken_prob)
        ent_attn = torch.mean(torch.unsqueeze(torch.stack(subtokens_in_ent), dim=1))
        attn_vector.append(ent_attn)
    attn_vector = torch.nn.functional.softmax(torch.stack(attn_vector))
    print("LAYER {}".format(l))
    print(attn_vector)
    output.append(attn_vector)
    print(torch.argmax(attn_vector))
torch.save(torch.stack(output), 'example_sentence.pt')