Section 4.5 Attacks on Imbalanced Data Distribution:
label_split: training data for GPT-2 has label >= 500 (training)
            test data 2376  (label 0-500)
            val data 500    (label 0-500)     
label_split_8cluster: We use Sentence-BERT to embed all persona sentences and perform k-means clustering on the embeddings to obtain 8 clusters.
