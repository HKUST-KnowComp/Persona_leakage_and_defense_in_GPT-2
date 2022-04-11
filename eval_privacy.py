'''
Used for val and test for attacker performance

external_model_path = sys.argv[3]
save_path = sys.argv[4]
Add new attacker to indicate type of external model

'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
import logging
import logging.handlers
from scipy.stats import entropy


logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
f_handler = logging.FileHandler(sys.argv[3]+'---case_study.log')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s -  %(message)s"))
logger.addHandler(f_handler)



torch.cuda.manual_seed_all(0)
np.random.seed(0)

processed_persona_path = config.processed_persona

external_criterion = nn.CrossEntropyLoss()


def get_processed_persona(kind,require_label = True):
    #processed_persona_path = config.processed_persona
    if(require_label):
        path = processed_persona_path + '/case_study.txt'
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

    all_logits = torch.cat(all_logits, dim=1)  #torch.Size([1, 611, 50257])

    # target
    all_logits = all_logits[:, :-1].contiguous()
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
    target_mask = torch.ones_like(target).float()
    
    loss = criterion(all_logits, target, target_mask, label_smoothing=0.02, reduce="batch")   
    loss /= num_gradients_accumulation

    lm_loss = loss
    if(max(labels) >= 0):
        #lm_loss += loss
        loss += ga_ratio * running_loss
        print('---label training with loss: ',running_loss)
    loss.backward()
        
    record_loss = lm_loss.item() * num_gradients_accumulation
    perplexity = np.exp(record_loss)
    #print(perplexity)
    #sys.exit(0)
    print('training loss: ', loss, 'PPL',perplexity)
    #if(perplexity == 1):
    #    print('***Sentence: ',dial_inputs)
    return record_loss, perplexity


def validate(dataloader,dataset):
    with torch.no_grad():
        pbar = progress_bar(dataloader)
        data_count, data_dist,uni_dist = get_data_dist(dataset,num_labels)

        total_ppl = []
        predict_num = 0 
        correct_num = 0
        y_true = []
        y_pred = []
        conv_list = []
        logits_list = []
        y_scores = []
        for batch in pbar:
            
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
            data_kl_list = []
            uni_kl_list = []
            loss_list = []
            running_loss = 0.0
            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                conv_dict = {}
                if role_ids[turn_num] == 0:
                    logits, past,hidden = model_A(dial_turn_inputs, past=past,output_hidden_states=True)
                    all_logits.append(logits)
                    
                else:
                    logits, past,hidden = model_B(dial_turn_inputs, past=past,output_hidden_states=True)
                    all_logits.append(logits)
                #now let's move to external module
                if(labels[turn_num] < 0):
                    labels[turn_num] = 0
                if(labels[turn_num] >= 0):
                    #make prediction here
                    predict_num += 1
                    hidden_out = torch.squeeze(hidden[-1], 0)
                    external_out = external_model(hidden_out)
                    est_dist = F.softmax(external_out)
                    est_squeeze = torch.squeeze(est_dist)
                    y_scores.append(est_squeeze.cpu().detach().numpy())
                    #y_scores.append()
                    #logit_cpu = external_out.cpu()
                    assert torch.sum(est_dist) - 1 <= 1e-5
                    assert external_out.ndim == 2
                    logits_list.append(est_dist)
                    #data_kl = entropy(data_dist, logit_cpu)
                    #uni_kl = entropy(uni_dist, logit_cpu)
                    data_kl = F.kl_div(data_dist.log(), est_dist).cpu()
                    uni_kl = F.kl_div(uni_dist.log(), est_dist).cpu()
                    data_kl_list.append(data_kl)
                    uni_kl_list.append(uni_kl)

                    predict_label = torch.argmax(external_out)
                    
                    label = torch.tensor([labels[turn_num]]).to(device)
                    #print('label: ',label)
                    #print('external_out: ',external_out.size())
                    external_criterion = nn.CrossEntropyLoss()
                    external_loss = external_criterion(external_out,label)
                    loss_list.append(external_loss.cpu())
                    #running_loss += external_loss
                    #print('external_loss passed with loss:',external_loss)
                    conv_dict['context'] = conv[0:turn_num+1]
                    conv_dict['predict'] = id2persona[predict_label.item()]
                    conv_dict['ground_truth'] = id2persona[labels[turn_num]]
                    conv_dict['correct_pred'] = 'False'
                    predict_label = int(predict_label.cpu())
                    label = int(labels[turn_num])
                    y_true.append(label)
                    y_pred.append(predict_label)
                    if(predict_label == label):
                        conv_dict['correct_pred'] = 'True'
                        correct_num += 1
                    conv_list.append(conv_dict)

            all_logits = torch.cat(all_logits, dim=1)
        
            # target
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()

            
            
            
            loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce="sentence")      

            
            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())
            

        logger.info(f"Epoch {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        with open(save_path, 'w') as f:
            json.dump(conv_list, f,indent=4)
        
        predict_num = len(y_pred)
        random_prediction = np.random.randint(4332, size=predict_num)
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"Epoch {ep} Validation prediction acc: {acc} over {predict_num} instances")


        random_acc = accuracy_score(y_true, random_prediction)
        logger.info(f"Epoch {ep} random prediction acc: {random_acc} over {predict_num} instances")

        f1 = f1_score(y_true, y_pred, average='weighted')
        logger.info(f"Epoch {ep} weighted f1 score: {f1}")

        random_f1 = f1_score(y_true, random_prediction, average='weighted')
        logger.info(f"Epoch {ep} weighted random_f1 score: {random_f1}")

        hist_bins = np.histogram(y_true, bins=np.arange(4332))
        bins = hist_bins[0]
        bins_key = np.arange(4332)
        bins_ratio = bins / predict_num
        bins_dict = {i:[j, k] for i, j, k in zip(bins_key, bins, bins_ratio)}
        index = np.argmax(bins_ratio)
        logger.info(f'max labels has id: {index} and ratio: {bins_dict[index]}')

        hist_bins = np.histogram(y_pred, bins=np.arange(4332))
        bins = hist_bins[0]
        bins_key = np.arange(4332)
        bins_ratio = bins / predict_num
        bins_dict = {i:[j, k] for i, j, k in zip(bins_key, bins, bins_ratio)}
        index = np.argmax(bins_ratio)
        logger.info(f'max predict labels has id: {index} and ratio: {bins_dict[index]}')

        logger.info(f'data_kl_list: {np.mean(data_kl_list)}')
        logger.info(f'uni_kl_list: {np.mean(uni_kl_list)}')
        logger.info(f'loss_list: {np.mean(loss_list)}')

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        label_space = [i for i in range(num_labels)]
        label_space = np.array(label_space)

        k_list= [2,3,4,5,10,50,100,500,1000,2000]
        for k in k_list:
            top_k = top_k_accuracy_score(y_true, y_scores, k=k,labels=label_space)
            logger.info(f'top {k} acc score: {top_k}')



        return np.mean(total_ppl)

def get_data_dist(dataset,num_labels):
    distribution = torch.zeros(num_labels).cuda()
    uni_dist = torch.ones(num_labels).cuda()
    uni_dist = uni_dist / num_labels
    for i,data in enumerate(dataset):
        labels = data['labels']
        for label in labels:
            if(label >= 0):
                distribution[label] += 1
    distribution_ratio = distribution / torch.sum(distribution)
    return distribution, distribution_ratio,uni_dist

chosen_persona_toid_path = processed_persona_path + '/persona2id.txt'
with open(chosen_persona_toid_path, 'r') as f:
    chosen_persona_toid = json.load(f)
'''
Cluster
0   Cars
1   Food & Drink
2   Animal & pet
3   Family
4   hobbies
5   works
6   personal info
7   music taste

'''
id2persona = {v: k for k, v in chosen_persona_toid.items()}

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
                            collate_fn=val_dataset.collate)
test_dataloader = DataLoader(dataset=test_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=test_dataset.collate)

logger.info('loading models')
path_postfix = 'GD_double'
#config.vocab_size = model_A_states["transformer.wte.weight"].shape[0]
model_A = AutoModelForCausalLM.from_pretrained(sys.argv[1])

model_B = AutoModelForCausalLM.from_pretrained(sys.argv[2])
external_model_path = sys.argv[3]
save_path = sys.argv[4] + '-----case_study'


device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)
attacker_type = sys.argv[5]
if '2layer' in attacker_type:
    external_model = persona_predict_model_2layer(out_num=num_labels)
elif 'trans' in attacker_type:
    external_model = persona_predict_model_transformer(out_num=num_labels)
else:
    logger.info('invalid attacker type')
    sys.exit(-1)

external_model.load_state_dict(torch.load(external_model_path))
external_model.to(device)



logger.info('loading_done')
criterion = SequenceCrossEntropyLoss()

# Optimizer
# define hyper-parameters
num_epochs = 2
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
no_decay = ['bias', 'ln', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)


scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100, 
                                            num_training_steps = num_train_optimization_steps)
update_count = 0
progress_bar = tqdm.tqdm_notebook
start = time.time()
old_ppl = -float('Inf')

for ep in range(num_epochs):

    #"Training"
    pbar = progress_bar(train_dataloader)
    #model_A.train()
    #model_B.train()
    
    for batch in pbar:

        batch = batch[0]

        if sum([len(item) for item in batch[1]]) > 1024:
            total_length = 0
            for index, item in enumerate(batch[1]):
                total_length = total_length + len(item)
                if total_length >= 1024:
                    batch = (batch[0][0:index-1], batch[1][0:index-1])
                    break
    
    #"Evaluation"
    model_A.eval()
    model_B.eval()
    external_model.eval()
    logger.info(external_model_path)
    #logger.info('-'*20,'Training','-'*20)
    #ppl_train = validate(train_dataloader)
    logger.info('-'*20+'Validation'+'-'*20)
    ppl_val = validate(val_dataloader,val_data)
    logger.info('-'*20+'Testing'+'-'*20)
    ppl_test = validate(test_dataloader,test_data)


    #Replace save with transformer save_pretrained


