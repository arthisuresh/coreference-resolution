import torch
import spacy
from pprint import pprint
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import pandas as pd
from collections import Counter
from collections import defaultdict
import numpy as np
import pickle

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
nlp = spacy.load('en_core_web_lg')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load file
data = pd.read_csv('/Users/arts/Desktop/cs224n/gap-coreference/gap-test.tsv', delimiter='\t')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

output_list = {'ID': [], 'A-coref': [], 'B-coref': []}
cells = torch.zeros((12, 12))
for i, row in data.iterrows():
    passage = row['Text']
    example_id = row['ID']
    output_list['ID'].append(example_id)
    processed = nlp(passage)
    sentences = [sent for sent in processed.sents]
    passage_with_separators = ' '.join(['[CLS]'] + [sent.text + ' [SEP]' for sent in processed.sents] + ['[SEP]'])
    passage_tokenized = tokenizer.tokenize(passage_with_separators)
    pronoun_offset = int(row['Pronoun-offset'])
    pronoun_index = 1
    pronoun = row['Pronoun']
    if pronoun_offset > 0:
        # finding index of pronoun in tokenized text
        passage_until_pronoun = passage[:pronoun_offset]
        passage_until_pronoun_processed = nlp(passage_until_pronoun)
        sentences_until_pronoun = [sent for sent in passage_until_pronoun_processed.sents]
        passage_until_pronoun_with_separators = ' '.join(['[CLS]'] + [sent.text + ' [SEP]' for sent in sentences_until_pronoun])
        if passage_until_pronoun_with_separators != passage_with_separators[:len(passage_until_pronoun_with_separators)]:
            passage_until_pronoun_with_separators = passage_until_pronoun_with_separators[:-5]
        # print(passage_until_pronoun_with_separators)
        passage_until_pronoun_tokenized = tokenizer.tokenize(passage_until_pronoun_with_separators)
        pronoun_indices = [i for i, t in enumerate(passage_tokenized) if t.lower() == pronoun.lower()]
        pronoun_index = [i for i in pronoun_indices if abs(i-len(passage_until_pronoun_tokenized)) < 3][0]
        pronoun_found = passage_tokenized[pronoun_index]
        assert pronoun.lower() == pronoun_found.lower()
    mentionA = row['A']
    mentionB = row['B']

    tokenized_ents = [tokenizer.tokenize(ent.text) for ent in processed.ents]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(passage_tokenized)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)

    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12

    n_layers = 12
    n_heads = 12
    output = []
    argmax_decisions = []
    ents = [ent.text.lower() for ent in processed.ents]
    unique_ents = list(set(ents))
    coalesced = np.zeros(len(unique_ents))
    attn_vectors = []
    for l in [8]:
        for h in [10]:
            ent_attns = []
            for i, ent in enumerate(ents):
                tokenized_ent = tokenized_ents[i]
                subtokens_in_ent = []
                for subtoken in tokenized_ent:
                    indexes_to_sum = [i for i, t in enumerate(passage_tokenized) if t == subtoken]
                    a_l = torch.squeeze(model.encoder.layer[l].attention.self.attention_scores)
                    a_lh = a_l[h, pronoun_index, :]
                    subtoken_prob = torch.sum(a_lh[indexes_to_sum])
                    subtokens_in_ent.append(subtoken_prob)
                ent_attn = torch.mean(torch.unsqueeze(torch.stack(subtokens_in_ent), dim=1))
                ent_attns.append(ent_attn)
            argmax = torch.argmax(torch.stack(ent_attns))[0]
            if ents[argmax] == mentionA.lower():
                output_list['A-coref'].append(True)
                output_list['B-coref'].append(False)
                if row['A-coref']:
                    cells[h, l] += 1
            elif ents[argmax] == mentionB.lower():
                output_list['A-coref'].append(False)
                output_list['B-coref'].append(True)
                if row['B-coref']:
                    cells[h, l] += 1
            else:
                output_list['A-coref'].append(False)
                output_list['B-coref'].append(False)
                if not (row['B-coref'] or row['A-coref']):
                    cells[h, l] += 1
    #     attn_vector = []
    #     for i, ent in enumerate(processed.ents):
    #         ent = ent.text
    #         tokenized_ent = tokenized_ents[i]
    #         subtokens_in_ent = []
    #         for subtoken in tokenized_ent:
    #             indexes_to_sum = [i for i, t in enumerate(passage_tokenized) if t == subtoken]
    #             k = torch.squeeze(model.encoder.layer[l].attention.self.attention_scores)
    #             s = torch.squeeze(torch.mean(k.narrow(1, pronoun_index, 1), dim=0))
    #             subtoken_prob = torch.sum(s[indexes_to_sum])
    #             subtokens_in_ent.append(subtoken_prob)
    #         ent_attn = torch.mean(torch.unsqueeze(torch.stack(subtokens_in_ent), dim=1))
    #         attn_vector.append(ent_attn)
    #     attn_vectors.append(torch.stack(attn_vector))
    # print(attn_vectors)
    # mean_attn = torch.mean(torch.stack(attn_vectors[:-6]), dim=0)
    # for i, ent in enumerate(processed.ents):
    #     unique_idx = unique_ents.index(ent.text.lower())
    #     coalesced[unique_idx] += mean_attn[i].item()
    # softmaxed_attn = torch.nn.functional.softmax(torch.Tensor(coalesced))
        
    # output.append(attn_vector)
    # argmax = torch.argmax(attn_vector)[0]
    # argmax_decisions.append(argmax)
    # softmaxes = list([x.item() for x in list(softmaxed_attn)])
    # ents = [ent.text for ent in processed.ents]
    # zip_together = dict(zip(unique_ents, softmaxes))
    # choose mode of all layers
    # argmax_decisions = Counter(argmax_decisions)
    # d = [k for k, v in argmax_decisions.items() if v == max(argmax_decisions.values())][0]
    # mentions_count = defaultdict(int)
    # for a in argmax_decisions:
    #     mentions_count[ents[a].lower()] += 1
    
    # populous_mention = max(zip_together, key=zip_together.get)
    # choose mode of last four layers
    # argmax_decisions = Counter(argmax_decisions)
    # print([ent for ent in processed.ents])
    # print(argmax_decisions)
    # d = [k for k, v in argmax_decisions.items() if v == max(argmax_decisions.values())][0]
    # choose last encoder layer
    # d = argmax_decisions[11]

#     print(passage)
#     if populous_mention == mentionA.lower():
#         output_list['A-coref'].append(True)
#         output_list['B-coref'].append(False)
#         print('A')
#     elif populous_mention == mentionB.lower():
#         output_list['A-coref'].append(False)
#         output_list['B-coref'].append(True)
#         print('B')
#     else:
#         output_list['A-coref'].append(False)
#         output_list['B-coref'].append(False)
#         print('Neither')
df = pd.DataFrame(output_list)
df.to_csv('attn_heuristic_h10_l8_test.tsv', header=False, index=False, sep='\t')
# cells = cells/len(data)
# pickle.dump(cells, open('attn_cells', 'wb'))