File name with no suffix: the original [PersonaChat dataset](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat) from ParlAI.

File name with suffix '_merge': aligned datasets by merging PersonaChat dataset with [DNLI dataset](https://arxiv.org/abs/1811.00671).

File name with suffix '_merge_shuffle': randomly shuffled dataset after merging.

File name with suffix '_merge_shuffle_8cluster': we use [Sentence-BERT](https://github.com/UKPLab/sentence-transformers) to embed all persona sentences and perform k-means clustering on the embeddings to obtain 8 clusters.

#### Section 4.5 Attacks on Imbalanced Data Distribution:
label_split: training data for GPT-2 has label_id >= 500 while test data has label_id 0-500.  
label_split_8cluster: training data for GPT-2 has label_id >= 3 while test data has label_id 0-2.
