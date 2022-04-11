import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2Config
#from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import sys
import json

torch.cuda.manual_seed_all(0)
np.random.seed(0)

processed_persona_path = config.processed_persona
freeze = True
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
        return role_ids, dial_tokens, labels
        
    def collate(self, unpacked_data):
        return unpacked_data

# Define loss
class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)

class persona_predict_model(nn.Module):
    def __init__(self, out_num, in_num=1024):
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

class persona_predict_model_2layer(nn.Module):
    def __init__(self, out_num, in_num=1024):
        super(persona_predict_model_2layer, self).__init__()
        hidden = 512
        self.fc1 = nn.Linear(in_num, hidden)
        self.fc2 = nn.Linear(hidden, out_num)
        self.relu = nn.ReLU()
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
        out = self.relu(out)
        out = self.fc2(out)
        #out = F.softmax(self.fc1(x),dim=1)

        return out

class persona_predict_model_transformer(nn.Module):
    def __init__(self, out_num, in_num=1024):
        super(persona_predict_model_transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)  #1024 as output
        self.fc1 = nn.Linear(1024, out_num)
        #self.fc1 = nn.Linear(in_num, out_num)
        #self.act = F.softmax()

    def forward(self, x, use_final_hidden_only = True):
        # x should be of shape (?,1024) according to gpt2 output
        x_resize = torch.unsqueeze(x, 0)
        
        # cut first dimension, now should of shape(1024) only
        # x = torch.squeeze(x, 0)
        #assert(x.size()[1] == out_shape)
        out = self.encoder_layer(x_resize)
        out_squeeze = torch.squeeze(out, 0)
        # use avg hidden
        x = torch.mean(x,dim=0,keepdim=True)
        final_out = self.fc1(x)
        #print(f'before encoder: {x_resize.size()}')
        #print(f'after encoder: {out.shape}')
        #print(f'after squeeze: {out_squeeze.shape}')
        #print(f'after fc1: {x.shape}')
        #print(f'final out: {final_out.shape}')
        return final_out


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce is "batch":
            # shape : scalar
            loss = loss.mean()

    return loss

# Training
def train_one_iter(external_model, batch, update_count, fp16=False, require_KL_loss=False, KL_ratio=0.5, ga_ratio=0.1):
    role_ids, dialog_tokens, labels = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

    past = None
    all_logits = []
    
    temp_sum = 0

    running_loss = 0.0
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        temp_sum += 1
        if role_ids[turn_num] == 0:

            #logits, past = model_A(dial_turn_inputs, past=past)
            logits, past, hidden = model_A(dial_turn_inputs, past=past,output_hidden_states=True)# hidden:torch.Size([1, 6, 1024])
            all_logits.append(logits)
            
        else:
            #logits, past = model_B(dial_turn_inputs, past=past)
            logits, past, hidden = model_B(dial_turn_inputs, past=past,output_hidden_states=True)
            all_logits.append(logits)

        #now let's move to external module
        if(labels[turn_num] >= 0):
            hidden_out = torch.squeeze(hidden[-1], 0)
            external_out = external_model(hidden_out)
            assert external_out.ndim == 2
            num_labels = external_out.size()[1]
            label = torch.tensor([labels[turn_num]]).to(device)
            #assign random label
            #label = np.random.randint(low=0,high=4332)
            #label = torch.tensor([label]).to(device)
            #print('label: ',label)
            #print('external_out: ',external_out.size())
            external_criterion = nn.CrossEntropyLoss()
            external_loss = external_criterion(external_out,label) 
            # gradient ascending use '-'
            running_loss += external_loss
            if(require_KL_loss):
                KL_loss = F.softmax(external_out,dim=1)
                KL_loss = torch.squeeze(KL_loss, 0)
                KL_loss = torch.log(KL_loss)
                KL_loss = torch.mean(KL_loss)
                # KL divergence with uniform distribution
                running_loss -= KL_ratio * KL_loss
            #print('external_loss passed with loss:',external_loss)

        # print('--'*20)
        # print('hidden:',len(hidden))
        # for i,h in enumerate(hidden):
        #     print(i,'-th hidden:',h.size())
        # if(temp_sum >= 5):
        #     sys.exit(0)
    all_logits = torch.cat(all_logits, dim=1)  #torch.Size([1, 611, 50257])
    #print(all_logits.size())
    # sys.exit(0)
    


    # target
    all_logits = all_logits[:, :-1].contiguous()
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
    target_mask = torch.ones_like(target).float()


    #print(all_logits.size())
    #print(target.size())
    
    loss = criterion(all_logits, target, target_mask, label_smoothing=0.02, reduce="batch")   
    loss /= num_gradients_accumulation
    record_loss = loss.item() * num_gradients_accumulation
    perplexity = np.exp(record_loss)
    print('training loss: ', loss, 'PPL',perplexity)
    #print('training loss: ', loss)
    #add external
    if(max(labels) >= 0):
        #lm_loss += loss

        loss += ga_ratio * running_loss
        print('---label training with loss: ',loss)
    loss.backward()
        


    return record_loss, perplexity


def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        total_ppl = []
        #total_ppl_recommender = []
        predict_num = 0 
        correct_num = 0

        for batch in pbar:
            if sum([len(item) for item in batch[0][1]]) > 1024:
                total_length = 0
                for index, item in enumerate(batch[0][1]):
                    total_length = total_length + len(item)
                    if total_length >= 1024:
                        batch = [(batch[0][0][0:index-1], batch[0][1][0:index-1])]
                        break
        
            role_ids, dialog_tokens, labels = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
            #dial_inputs_rec = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens if item[0] == 32]
            past = None
            all_logits = []
            #all_logits_rec = []
            running_loss = 0.0
            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    #logits, past = model_A(dial_turn_inputs, past=past)
                    logits, past,hidden = model_A(dial_turn_inputs, past=past,output_hidden_states=True)
                    all_logits.append(logits)
                    
                    #all_logits_rec.append(logits)
                else:
                    #logits, past = model_B(dial_turn_inputs, past=past)
                    logits, past,hidden = model_B(dial_turn_inputs, past=past,output_hidden_states=True)
                    all_logits.append(logits)
                #now let's move to external module
                if(labels[turn_num] >= 0):
                    #make prediction here
                    predict_num += 1
                    hidden_out = torch.squeeze(hidden[-1], 0)
                    external_out = external_model(hidden_out)
                    assert external_out.ndim == 2
                    predict_label = torch.argmax(external_out)
                    
                    label = torch.tensor([labels[turn_num]]).to(device)
                    #print('label: ',label)
                    #print('external_out: ',external_out.size())
                    #external_criterion = nn.CrossEntropyLoss()
                    #external_loss = external_criterion(external_out,label)
                    #running_loss += external_loss
                    #print('external_loss passed with loss:',external_loss)
                    if(predict_label == label):
                        correct_num += 1

            all_logits = torch.cat(all_logits, dim=1)
            #all_logits_rec = torch.cat(all_logits_rec, dim=1)
            



            # target
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()
            
            
            loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce="sentence")      

            
            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())
            
            #ppl_recommender = torch.exp(loss_recommender)
            #total_ppl_recommender.extend(ppl_recommender.tolist())

        print(f"Epcoh {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        if(max(labels) >= 0):
            acc = correct_num / predict_num
            print(f"Epcoh {ep} Validation prediction loss: {acc}")
        #print(f"Epcoh {ep} Validation Perplexity on recommender (A): {np.mean(total_ppl_recommender)} Variance: {np.var(total_ppl_recommender)}")
        
        return np.mean(total_ppl)



chosen_persona_toid_path = processed_persona_path + '/persona2id.txt'
with open(chosen_persona_toid_path, 'r') as f:
    chosen_persona_toid = json.load(f)
num_labels = len(chosen_persona_toid)

train_data = get_processed_persona('train')
val_data = get_processed_persona('dev')

batch_size = 1



tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

train_dataset = PersonaDataset(train_data, tokenizer)
val_dataset = PersonaDataset(val_data, tokenizer)
train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=train_dataset.collate)
val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)


print('loading models')
#config.vocab_size = model_A_states["transformer.wte.weight"].shape[0]
model_A = AutoModelForCausalLM.from_pretrained("model_dialogpt_fc_lastonly_0.1ga_10KL_1attack_shuffle_newext_noKL")

model_B = AutoModelForCausalLM.from_pretrained("model_dialogpt_fc_lastonly_0.1ga_10KL_1attack_shuffle_newext_noKL")


device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)


external_model = persona_predict_model_2layer(out_num=num_labels).to(device)

print('loading_done')
criterion = SequenceCrossEntropyLoss()

# Optimizer
# define hyper-parameters
num_epochs = 10
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

#param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
#param_optimizer = list(external_model.parameters())
#no_decay = ['bias', 'ln', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]

optimizer = AdamW(external_model.parameters(), 
                  lr=3e-5,
                  eps=1e-06)

#scheduler = WarmupLinearSchedule(optimizer,
#                                 warmup_steps=100,
#                                 t_total=num_train_optimization_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100, 
                                            num_training_steps = num_train_optimization_steps)
update_count = 0
progress_bar = tqdm.tqdm_notebook
start = time.time()
old_ppl = -float('Inf')

if freeze:
    ####freeze model_A & B
    for i,param in enumerate(model_A.base_model.parameters()):
        if(i==0):
            continue
        param.requires_grad = False
    for i,param in enumerate(model_B.base_model.parameters()):
        if(i==0):
            continue
        param.requires_grad = False

for ep in range(num_epochs):

    #"Training"
    pbar = progress_bar(train_dataloader)
    if freeze:
        model_A.eval()
        model_B.eval()
    else:
        model_A.train()
        model_B.train()
    
    for batch in pbar:

        batch = batch[0]

        if sum([len(item) for item in batch[1]]) > 1024:
            total_length = 0
            for index, item in enumerate(batch[1]):
                total_length = total_length + len(item)
                if total_length >= 1024:
                    batch = (batch[0][0:index-1], batch[1][0:index-1])
                    break
    
        record_loss, perplexity = train_one_iter(external_model,batch, update_count, fp16=False)
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            # update for gradient accumulation
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end
            
            # show progress
            pbar.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)

    #"Evaluation"
    model_A.eval()
    model_B.eval()
    ppl = validate(val_dataloader)
    print('val result ppl:',ppl)

    #Replace save with transformer save_pretrained
    #GA+KL/single/model_dialogpt_fc_lastonly_0.1ga_10KL_shuffle_noGAloss
    #model_A.save_pretrained("GD_double/model_A")
    #model_B.save_pretrained("GD_double/model_B")
    if freeze:
        torch.save(external_model.state_dict(), "attackers/attacker_2layer_0.1ga_10KL_1attack_shuffle_newext_noKL")
    else:
        torch.save(external_model.state_dict(), "GA_double_GA_ratio/external_model_dialogpt_fc_lastonly_0.1ga_0.5KL_shuffle_GD_nofreeze")
        model_A.save_pretrained("GA_double_GA_ratio/model_A_dialogpt_fc_lastonly_0.1ga_0.5KL_shuffle_nofreeze")
        model_A.save_pretrained("GA_double_GA_ratio/model_B_dialogpt_fc_lastonly_0.1ga_0.5KL_shuffle_nofreeze")

