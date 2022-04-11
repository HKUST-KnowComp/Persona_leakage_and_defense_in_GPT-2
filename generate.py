'''
Used for generating response for testing data
Do not require attacker model
model_A_path = sys.argv[1]
model_B_path = sys.argv[2]
save_path = sys.argv[3]
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# import huggingface transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import sys
import json
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

progress_bar = tqdm.tqdm_notebook
processed_persona_path = config.processed_persona

class persona_predict_model(nn.Module):
    def __init__(self, in_num=1024, out_num=8):
        super(persona_predict_model, self).__init__()
        self.fc1 = nn.Linear(in_num, out_num)
        #self.act = F.softmax()

    def forward(self, x, use_final_hidden_only = True):
        # x should be of shape (?,1024) according to gpt2 output
        out_shape = x.size()[-1]
        if(use_final_hidden_only):
            # avg the info 
            x = torch.unsqueeze(x[-1],0)
        else:
            x = torch.mean(x,dim=0,keepdim=True)
        # cut first dimension, now should of shape(1024) only
        # x = torch.squeeze(x, 0)
        assert(x.size()[1] == out_shape)
        out = self.fc1(x)
        #out = F.softmax(self.fc1(x),dim=1)

        return out


def get_processed_persona(kind,require_label = True):
    #processed_persona_path = config.processed_persona
    if(require_label):
        path = processed_persona_path + '/%s_merged_shuffle.txt' % kind
    else:
        path = processed_persona_path + '/%s.txt' % kind
    with open(path, 'r') as f:
        data = json.load(f)
    return data

class PersonaDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.turn_ending = tokenizer.encode('<|endoftext|>')  # dialogpt pretrain approach
        #self.turn_ending = [628, 198] #628:\n\n 198:\n
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        conv = self.data[index]['conv']
        dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in conv]
        role_ids = [0,1] * (len(conv)//2)

        #for labels
        labels = self.data[index]['labels']

        assert len(role_ids) == len(dial_tokens)
        return role_ids, dial_tokens, labels, conv
        
    def collate(self, unpacked_data):
        return unpacked_data



def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits

def generate_sentence(logits,past):
    sent = []
    prev_input = None
    logits = logits[:, -1, :] / temperature
    logits = top_filtering(logits, top_k=top_k, top_p=top_p)

    probs = torch.softmax(logits, dim=-1)

    prev_input = torch.multinomial(probs, num_samples=1)
    prev_word = prev_input.item()
    #if prev_word == eos[0]:
    #    break
    sent.append(prev_word)

    for i in range(500):
        logits, past = model_B(prev_input, past=past)
        logits = logits[:, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)

        probs = torch.softmax(logits, dim=-1)

        prev_input = torch.multinomial(probs, num_samples=1)
        prev_word = prev_input.item()

        if prev_word == eos[0]:
            break
        sent.append(prev_word)
    
    output = tokenizer.decode(sent)

    return output


def generate_response(dataloader, all_pred = False):
    with torch.no_grad():
        json_list = []

        pbar = progress_bar(dataloader)

        total_ppl = []
        predict_num = 0 
        correct_num = 0

        y_true = []
        y_pred = []
        for batch in pbar:
            ref_dict = {}
            ref_gt = []
            ref_output = []
            if sum([len(item) for item in batch[0][1]]) > 1024:
                total_length = 0
                for index, item in enumerate(batch[0][1]):
                    total_length = total_length + len(item)
                    if total_length >= 1024:
                        batch = [(batch[0][0][0:index-1], batch[0][1][0:index-1])]
                        break
        
            role_ids, dialog_tokens, labels, conv = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
            
            past = None
            all_logits = []
            running_loss = 0.0
            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:

                    logits, past,hidden = model_A(dial_turn_inputs, past=past,output_hidden_states=True)
                    all_logits.append(logits)
                    generated_output = generate_sentence(logits,past)
                    ref_output.append(generated_output)
                else:
                    logits, past,hidden = model_B(dial_turn_inputs, past=past,output_hidden_states=True)
                    all_logits.append(logits)
                    ref_gt.append(conv[turn_num])

            ref_dict['conv'] = conv
            ref_dict['output'] = ref_output
            ref_dict['ref'] = ref_gt
            json_list.append(ref_dict)
        with open(save_path, 'w') as f:
            json.dump(json_list, f,indent=4)

            #dump dict

save_path = ''
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
print('tokenizer done')
#model_A_path = 'GA_double_KL/model_A_dialogpt_fc_lastonly_0.1ga_10KL_shuffle_noGA'
#model_B_path = 'GA_double_KL/model_B_dialogpt_fc_lastonly_0.1ga_10KL_shuffle_noGA'
model_A_path = sys.argv[1]
model_B_path = sys.argv[2]
save_path = sys.argv[3]
model_A =AutoModelForCausalLM.from_pretrained(model_A_path)
model_B =AutoModelForCausalLM.from_pretrained(model_B_path)
print('model done')
device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)
model_A.eval()
model_B.eval()
print('model loaded to GPU')

# beg huggingface not to change this anymore
#model.lm_head.weight.data = model.transformer.wte.weight.data

eos = [tokenizer.encoder["<|endoftext|>"]]

past = None
temperature = 0.9
top_k = -1
top_p = 0.9


prev_input = None


chosen_persona_toid_path = processed_persona_path + '/persona2id.txt'
with open(chosen_persona_toid_path, 'r') as f:
    chosen_persona_toid = json.load(f)
num_labels = len(chosen_persona_toid)

train_data = get_processed_persona('train')
val_data = get_processed_persona('dev')
test_data = get_processed_persona('test')

batch_size = 1



tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

train_dataset = PersonaDataset(train_data, tokenizer)
val_dataset = PersonaDataset(val_data, tokenizer)
test_dataset = PersonaDataset(test_data, tokenizer)


train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=train_dataset.collate)
val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)
test_dataloader = DataLoader(dataset=test_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)

num_epochs = 1
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

print('loading done')
#"Evaluation"
model_A.eval()
model_B.eval()
#print('-'*20,'Training','-'*20)
#ppl_train = validate(train_dataloader)
#print('-'*20,'Validation','-'*20)
#ppl_val = validate(val_dataloader)
print('-'*20,'Testing','-'*20)
generate_response(test_dataloader)
