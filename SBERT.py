import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(filename='/home/iailab31/mirhajm0/results/sbert.log', encoding='utf-8', level=logging.INFO)

def return_doc(doc):
    return doc

# Load dataset
df_train_raw = pd.read_json("/home/iailab31/mirhajm0/data/trn.json", lines=True)
df_test_raw = pd.read_json("/home/iailab31/mirhajm0/data/tst.json", lines=True)
y = pd.read_csv('/home/iailab31/mirhajm0/data/Yf.txt', header=None, encoding='latin-1')
logging.info("data is loaded")



# Add the target labels to the train data
df_test = df_test_raw.copy()
target_index = df_test.columns.get_loc("target_ind")
df_test.insert(target_index+1, "target_label", np.zeros)
topic = []
for values in df_test.target_ind:
  label = [y.loc[int(i)][0] for i in values]
  topic.append(label)
df_test.target_label = topic
logging.info("data is ready")

# TF-IDF weighting for the labels
corpus = df_train.target_label.values.tolist()
vectorizer = TfidfVectorizer(analyzer='word', tokenizer=return_doc, preprocessor=return_doc, token_pattern=None)
X = vectorizer.fit_transform(corpus)
get_feature_names_out = vectorizer.get_feature_names_out()


# Getting the mean weights and adding it to the dataset
mean_val = {}
for i in range(len(get_feature_names_out)):
  mean_val[get_feature_names_out[i]] = np.mean(X[:, i].toarray())
# Add the TF_idf_weights_cat to the dataset
target_index = df_train.columns.get_loc("target_label")
try:
  df_train.insert(target_index+1, "TF_idf_weights_cat", np.zeros)
except:
  pass
TF_idf_weights_cat = []
for values in df_train.target_label:
  weights = [mean_val[i] for i in values]
  TF_idf_weights_cat.append(weights)
df_train.TF_idf_weights_cat = TF_idf_weights_cat
logging.info("TF-IDF weighting is done.")

# Create characteristic vectors with transactionencoder and Compute df_shared_cat for topics
te = TransactionEncoder()
te_ary = te.fit(topic).transform(topic)
TF = pd.DataFrame(te_ary, columns=te.columns_)
TF = TF.T
df_shared_cat = TF.T.astype(int).dot(TF.astype(int))
np.fill_diagonal(df_shared_cat.values, -1)
values, counts = np.unique(df_shared_cat.values, return_counts=True)

# x is the index and y is the columns for specific values in shared_cats
x, y = np.asarray(df_shared_cat.values==0).nonzero()
ind = np.array(list(zip(x,y)))
logging.info("Charecteristic vectors are ready")
# downsampeling
ind = ind[np.random.choice(ind.shape[0], size=(ind.shape[0])//10**3, replace=False), :]


# Cosine similarity computation
union_xy = list(set(ind[:, 0]).union(set(ind[:,1])))
doc_emb = np.array(TF[union_xy].T.astype(int).values.tolist())
cos_doc = cosine_similarity(doc_emb, Y=None, dense_output=True)
df_cos_doc = pd.DataFrame(cos_doc, columns=TF[union_xy].columns, index=TF[union_xy].columns)

# Weighted cosine similarity computation
weighted_characteristic_matrix = pd.DataFrame(X.toarray()).T
weighted_characteristic_matrix.index = TF.index
weighted_doc_emb = np.array(weighted_characteristic_matrix[union_xy].T.values.tolist())
weighted_cos_doc = cosine_similarity(weighted_doc_emb, Y=None, dense_output=True)
df_weighted_cos_doc = pd.DataFrame(weighted_cos_doc, columns=TF[union_xy].columns, index=TF[union_xy].columns)
logging.info("cosine similarity is computed")
# Fine-tuning with unweighted cosine similarities
model = SentenceTransformer('all-MiniLM-L6-v2')
train_examples = []
for i in range(len(df_weighted_cos_doc.columns)):
  for j in range(i, len(df_weighted_cos_doc.columns), 1):
    train_examples.append(InputExample(texts = [df_train.content[df_weighted_cos_doc.columns[i]],df_train.content[df_weighted_cos_doc.columns[j]]], label = float(df_weighted_cos_doc.iloc[i,j])))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=20)
X_train = model.encode(list(df_train.content))
X_test = model.encode(list(df_test.content) )
# Save output
np.savetxt('/home/iailab31/mirhajm0/results/train_embeddings.txt', X_train)
#x = np.loadtxt('train_embeddings.txt')

np.savetxt('/home/iailab31/mirhajm0/results/test_embeddings.txt', X_test)
#x = np.loadtxt('train_embeddings.txt')
logging.info('model is fine-tuned')
# Fine-tuning with weighted cosine similarities
train_examples_weighted = []
for i in range(len(df_weighted_cos_doc.columns)):
  for j in range(i, len(df_weighted_cos_doc.columns), 1):
    train_examples_weighted.append(InputExample(texts = [df_train.content[df_weighted_cos_doc.columns[i]],df_train.content[df_weighted_cos_doc.columns[j]]], label = float(df_weighted_cos_doc.iloc[i,j])))

train_dataloader_weighted = DataLoader(train_examples_weighted, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader_weighted, train_loss)], epochs=1, warmup_steps=20)


X_train_weighted = model.encode(list(df_train.content))
X_test_weighted = model.encode(list(df_test.content) )

# Save output
np.savetxt('/home/iailab31/mirhajm0/results/train_embeddings_weighted.txt', X_train_weighted)
#x = np.loadtxt('train_embeddings.txt')

np.savetxt('/home/iailab31/mirhajm0/results/test_embeddings_weighted.txt', X_test_weighted)
#x = np.loadtxt('train_embeddings.txt')
logging.info('model is fine-tuned')
