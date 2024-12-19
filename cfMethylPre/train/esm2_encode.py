import torch
import esm
import pandas as pd

if not torch.cuda.is_available():
    print("No GPU found. Please make sure a GPU is available.")
    exit()

device = torch.device('cuda:0')

model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval() 

probe_seq = pd.read_csv('../result/GNN_encode/probe_seq_all.csv')
data = list(zip(probe_seq['IlmnID'], probe_seq['SourceSeq']))

batch_size = 1


sequence_representations = []
for i in range(0, len(data), batch_size):
    batch_data = data[i:i+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[48], return_contacts=True)
    token_representations = results["representations"][48]

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).tolist())

columns = ['representation_{}'.format(i) for i in range(len(sequence_representations[0]))]

result_df = pd.DataFrame(sequence_representations, columns=columns)
result_df['IlmnID'] = probe_seq['IlmnID']
=======
import torch
import esm
import pandas as pd

if not torch.cuda.is_available():
    print("No GPU found. Please make sure a GPU is available.")
    exit()

device = torch.device('cuda:0')

model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval() 

probe_seq = pd.read_csv('../result/GNN_encode/probe_seq_all.csv')
data = list(zip(probe_seq['IlmnID'], probe_seq['SourceSeq']))

batch_size = 1


sequence_representations = []
for i in range(0, len(data), batch_size):
    batch_data = data[i:i+batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[48], return_contacts=True)
    token_representations = results["representations"][48]

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).tolist())

columns = ['representation_{}'.format(i) for i in range(len(sequence_representations[0]))]

result_df = pd.DataFrame(sequence_representations, columns=columns)
result_df['IlmnID'] = probe_seq['IlmnID']
result_df.to_csv('../data/encode_matrix/t48_15B_all.csv', index=False)
