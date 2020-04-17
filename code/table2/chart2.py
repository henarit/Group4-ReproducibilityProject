
import os, sys, random, yaml, gc
from itertools import *

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

from utils import *
from models import TrainableModel
from modules import Highway
from generators import AbstractPatientGenerator
import fetch

import IPython

#For gene and clinical data

# ==== OLD: ====
# MULTIMODAL = sys.argv[1]
# ==== NEW: ====
# This checks if multimodal dropout keyword is specified 
MULTIMODAL = 'no_multi'
if len(sys.argv) > 1:
  MULTIMODAL = '' + sys.argv[1]
# == END NEW ==

OUTFILE = f"results/{MULTIMODAL}_gene_clinical.txt"
OUTPKL = f"results/{MULTIMODAL}_gene_clinical.pkl"
print (OUTFILE)

# ==== NEW: ====
# use GPU to speed up
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
# == END NEW ==

class Network(TrainableModel):

    def __init__(self):
        super(Network, self).__init__()

        self.fcm = nn.Linear(1881, 256)
        # ==== OLD ====
        # self.fcc = nn.Linear(7, 256)
        # ==== NEW ====
        # the input dimension = nr of clinical variables
        self.fcc = nn.Linear(4, 256)
        # == END NEW ==
        self.fcg = nn.Linear(60483, 256)
        self.highway = Highway(256, 10, f=F.relu)
        self.fc2 = nn.Linear(256, 2)
        self.fcd = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1, affine=True)

    def forward(self, data, mask):

        x = data['gene']
        x = x.view(x.shape[0], -1)
        x = F.dropout(x, 0.4)
        # x = F.tanh(self.fcm(x)) # ... deprecated
        # ==== NEW ====
        x = torch.tanh(self.fcg(x))
        # == END NEW ==

        y = data['clinical']
        y = y.view(y.shape[0], -1)
        # y = F.tanh(self.fcc(y)) # ... deprecated
        # ==== NEW ====
        y = torch.tanh(self.fcc(y))
        # == END NEW ==

        mean = masked_mean((x, y), (mask["gene"], mask["clinical"]))

        var = masked_variance((x, y), (mask["gene"], mask["clinical"])).mean()
        var2 = masked_mean (((x - mean.mean())**2, (y - mean.mean())**2), \
                            (mask["gene"], mask["clinical"]))

        ratios = var/var2.mean(dim=1)
        ratio = ratios.clamp(min=0.02, max=1.0).mean()

        x = mean

        x = self.bn1(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.highway(x)
        x = self.bn2(x)

        score = F.log_softmax(self.fc2(x), dim=1)
        hazard = self.fcd(x)

        return {"score": score, "hazard": hazard, "ratio": ratio.unsqueeze(0)}

    def loss(self, pred, target):

        # Arrays with true/target values for the batch
        vital_status = target["vital_status"]
        days_to_death = target["days_to_death"]

        # Array with predictions for the batch
        hazard = pred["hazard"].squeeze()

        # pred["score"] is an array where every entry is log-probabilities
        # for two classes. nll_loss = negative log likelihood loss.
        # the Comparison to vital_status suggesrs that "score" should be 
        # interpreted as log-probabilities of patient being dead or alive?
        loss = F.nll_loss(pred["score"], vital_status)
        
        # sort the patients in the batch by their days_to_death in ascending order.
        # idx is the list of indexes corresponding to this sorting
        _, idx = torch.sort(days_to_death)

        # ==== OLD ====
        # hazard_probs = F.softmax(hazard[idx].squeeze()[1-vital_status.byte()])

        # ==== NEW ====
        # dead_patients is a bool array indicating true for dead patients in batch
        # 1-vital_status is a list indicating which patients are dead or alive
        # vital_status[i] = 1 if patient i in batch is ALIVE
        dead_patients = [bool(is_dead_integer) for is_dead_integer in 1-vital_status]
        
        # Take the predicted 'hazard' for patients we know are DEAD
        # (patients that are still alive are ignored) 
        # length of array: number of dead patients in this batch
        # the array is sorted (ascending) according to the TRUE days_to_death
        hazard_dead_only = hazard[idx[dead_patients]]

        # Convert the hazards to "probabilities" using softmax
        hazard_probs = F.softmax(hazard_dead_only, dim=0)
        # == END NEW ==

        N = torch.tensor(hazard_probs.shape[0]) # 1 = Nr of dead patients + 1
        
        # ==== OLD ====
        # weights_cum = torch.range(1, N) # = [1, 2, ... , N]
        # torch.range is soon deprecated
        # ==== NEW ====
        weights_cum = torch.arange(1, N+1) # = [1, 2, ... , N]
        
        # Use GPU if available
        if use_cuda:
          N = N.cuda()
          weights_cum = weights_cum.cuda()
        # == END NEW ==

        # w1 = [1, 2, ... , N], w2 = [N-1, ... 1, 0]
        w1, w2 = weights_cum, N - weights_cum

        # cumulative sum of the hazard_probs, starting with 0
        # cumulative order: ascending days_to_death
        # ==== NEW ====
        if use_cuda: zero_tensor = torch.tensor(0.0).cuda()
        else: zero_tensor = torch.tensor(0.0)
        # == END NEW ==
        hazard_cum = torch.stack([zero_tensor] + list(accumulate(hazard_probs)))

        # p = cumulative sum of the hazard_probs (not starting with 0)
        # q = 1-p 
        p, q = hazard_cum[1:], 1-hazard_cum[:-1]

        # stack the columns p and q beside each other
        probs = torch.stack([p, q], dim=1) # shape: (nr_dead_patients, 2)

        # convert probability --> log probability
        logits = torch.log(probs)

        # ==== OLD ====
        # ll1 = (F.nll_loss(logits, torch.zeros(N).long(), reduce=False) * w1)/N
        # ll2 = (F.nll_loss(logits, torch.ones(N).long(), reduce=False) * w2)/N
        # ==== NEW ==== 
        # reduce=False deprecated ... use reduction='none' instead
        # Test: the following return the exact same thing:
        #
        #   print(-logits[:,0])
        #   print(F.nll_loss(logits, torch.zeros(N).long(), reduction='none'))
        # 
        # i.e. we slice the array. Why do it this way?

        neg_logits_dead = -logits[:,0]
        neg_logits_alive = -logits[:,1]

        ll1 = neg_logits_dead * w1 / N
        ll2 = neg_logits_alive * w2 / N

        # == END NEW ==

        loss2 = torch.mean(ll1 + ll2)
        loss3 = pred["ratio"].mean()

        return loss + loss2 + loss3*0.3

    def score(self, pred, target):
        #R, p = pearsonr(pred, target)
        vital_status = target["vital_status"]
        days_to_death = target["days_to_death"]
        score_pred = pred["score"][:, 1]
        hazard = pred["hazard"][:, 0]

        auc = roc_auc_score(vital_status, score_pred)
        cscore = concordance_index(days_to_death, -hazard, np.logical_not(vital_status))

        return {"AUC": auc, "C-index": cscore, "Ratio": pred["ratio"].mean()}


class DataGenerator(AbstractPatientGenerator):

    def __init_(self, **kwargs):
        super(DataGenerator, self).__init__(**kwargs)

    def sample(self, case, mode='train'):

        mirna_data = fetch.mirna_data(case)

        if mirna_data is not None:
            mirna_data = torch.tensor(mirna_data).float()

        gene_data = fetch.gene_data(case)

        if gene_data is not None:
            gene_data = torch.tensor(gene_data).float()

        clinical_data = fetch.clinical_data_expanded(case)
        
        if clinical_data is not None:
            clinical_data = torch.tensor(clinical_data).float()

        vital_status = fetch.vital_status(case)
        days_to_death = fetch.days_to_death(case)
        if days_to_death is False or days_to_death is None:
            vital_status = True
            days_to_death = 20000
        
        if MULTIMODAL == 'multi':
            if mode == 'train' and random.randint(1, 4) == 1: clinical_data = None
            if mode == 'train' and random.randint(1, 4) == 1: mirna_data = None
            if mode == 'train' and random.randint(1, 4) == 1: gene_data = None

        if clinical_data is None and gene_data is None: return None
        if vital_status is None: return None

        return {"clinical": clinical_data, "gene": gene_data},\
                {"vital_status": torch.tensor(vital_status).long(),
                "days_to_death": torch.tensor(days_to_death).float()}
    

if __name__ == "__main__":

  # Dictionary of lists containing C-index scores
  c_index_dict = {}
  for f in fetch.disease_lookup:
    if f is not None:
      c_index_dict[f] = []

  for i in range(10):
    print('------ iteration ', i, '------')

    model = Network()
    model.compile(optim.Adam, lr=8e-4)

    datagen = DataGenerator(samples=40000, val_samples=10000)

    stratified = {}
    for case in datagen.train_cases:
        cancer_type = fetch.cancer_type(case)
        cancer_type = fetch.disease_lookup[cancer_type]
        if cancer_type is None: continue
        stratified.setdefault(cancer_type, []).append(case)

    for epoch in range(0, 5):
      print('epoch ' + str(epoch)) 
      train_data = batched(datagen.data(mode='train'), batch_size=64)
      val_data = batched(datagen.data(mode='val', cases=datagen.val_cases), batch_size=64)
      model.fit(train_data, validation=val_data, verbose=True)

    for cancer_type, cases in sorted(stratified.items()):
        test_data = batched(datagen.data(mode='val', cases=cases), batch_size=64)
        try:
            score = model.eval_data(test_data)
        except:
            score = 0.0
        
        c_index_dict[cancer_type].append(score)

  # Pickle and save the results
  with open(OUTPKL, 'wb') as handle:
      pickle.dump(c_index_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print('OBS Saving model for epoch ' + str(epoch) + 'and iteration ' + str(i))
  model.save(f"results/predict2_{MULTIMODAL}.pth")
  
  with open(OUTFILE, "w") as outfile:
    print('====== average C-index scores =======')
    print('====== average C-index scores =======', file=outfile)

    for cancer_type, scores in sorted(c_index_dict.items()):
      print(f"{cancer_type}, {sum(scores)/len(scores)}")
      print(f"{cancer_type}, {sum(scores)/len(scores)}", file=outfile)
        




