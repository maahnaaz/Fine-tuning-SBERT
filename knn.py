import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pickle
import math
from scipy.sparse import lil_matrix
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
from sklearn.metrics import precision_recall_fscore_support, ndcg_score, precision_score, recall_score, f1_score, distance, accuracy_score


logging.basicConfig(filename='/home/iailab31/mirhajm0/results/sbert.log', level=logging.INFO, format='%(asctime)s - %(levelname)-8s %(message)s')
logger = logging.getLogger()


NUM_LABELS =13330
def create_label_matrix(idx, Y):
  global NUM_LABELS
  arr = np.zeros((len(idx), NUM_LABELS))
  for k, i in enumerate(Y[idx]):
    for j in i:
        arr[k, j] = 1
  return arr
  

def TopK(y_p, k):
  df = pd.DataFrame(y_p)
  s = df.sum()
  df = df[s.sort_values(ascending=False).index[:k]]
  df_1 = df.copy()
  for i in range(len(df_1)):
    for j in df_1.columns:
      if df_1.iloc[i][j] == 1:
         df_1.loc[i, j] = int(j)
  y_p = df_1.values

  return y_p, df.columns.tolist()
  
def precision_at_k_score(y_test, y_test1, y_pred, k=5):
  y_p, frequent = TopK(y_pred, k)
  score = []
  y_test = y_test.tolist()
  for i in range(len(y_test)):
    shared = list(set(y_test[i]).intersection(set(frequent)))
    if  len(shared) != 0:
    	common_values = set(y_test[i]).intersection(set(y_p[i, :]))
    	if np.count_nonzero(y_p[i, :]) == 0:
    		score.append(0)
    	else:
    		score.append(len(common_values)/np.count_nonzero(y_p[i, :]))

  return np.mean(np.array(score))

# Load the data     
df_train = pickle.load(open("/home/iailab31/mirhajm0/results/df_train.p","rb"))
df_test = pickle.load(open("/home/iailab31/mirhajm0/results/df_test.p","rb"))
X_TF = pickle.load(open("/home/iailab31/mirhajm0/results/X_TF.p","rb"))
logging.info("Data is loaded.")

"""
saved_model = torch.load('/home/iailab31/mirhajm0/results/fine_tune_model')
X_train = saved_model.encode(list(df_train.content))
X_test = saved_model.encode(list(df_test.content))
logging.info("Embedings are computed.")

pickle.dump(X_train, open("/home/iailab31/mirhajm0/results/train_embeddings1.p", "wb" ))
pickle.dump(X_test, open("/home/iailab31/mirhajm0/results/test_embeddings.p", "wb" ))
logging.info("Embeddings are saved.")
"""


# Load the embeddings 
X_train = pickle.load(open("/home/iailab31/mirhajm0/results/train_embeddings1.p","rb"))
X_test = pickle.load(open("/home/iailab31/mirhajm0/results/test_embeddings.p","rb"))
logging.info("Embeddings are loaded.")


X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
y_train = df_train["target_ind"]
y_test = df_test["target_ind"]
n_test = 15000
idx_test = np.random.choice(len(X_test), size=n_test, replace=False)
n_train = 350000
idx_train = np.random.choice(len(X_train), size=n_train, replace=False)

#pickle.dump(idx_test, open("/home/iailab31/mirhajm0/results/idx_test.p", "wb" ))
#pickle.dump(idx_train, open("/home/iailab31/mirhajm0/results/idx_train.p", "wb" ))
y_train1 = create_label_matrix(idx_train, y_train)
y_test1 = create_label_matrix(idx_test, y_test)
#knn = KNeighborsClassifier()
#k_range = list(range(1, 4))
#param_grid = dict(n_neighbors=k_range)
# defining parameter range
#grid = GridSearchCV(knn, param_grid, cv=2, scoring='accuracy', verbose=1)
#grid_search = grid.fit(X_train[idx_train], MultiLabelBinarizer().fit_transform(y_train[idx_train]))
#knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1, algorithm='kd_tree')
knn.fit(X_train[idx_train], y_train1)
logging.info("knn is fit.")

# Prediction
y_pred = knn.predict(X_test[idx_test])

	
pickle.dump(y_pred, open("/home/iailab31/mirhajm0/results/y_pred_n3.p","wb"))
logging.info("Prediction is done.")
#y_pred = pickle.load(open("/home/iailab31/mirhajm0/results/y_pred_n6.p","rb"))
acc = accuracy_score(y_test1, y_pred)
logging.info("accuracy is "+ str(acc))
precision_3 = precision_at_k_score(y_test[idx_test], y_test1, y_pred, k=1)
precision_5 = precision_at_k_score(y_test[idx_test], y_test1, y_pred, k=5)
ndcg_3 = ndcg_score(y_test1, y_pred, k=1)
ndcg_5 = ndcg_score(y_test1, y_pred, k=5)
logging.info("precision_3 is "+ str(precision_3))
logging.info("precision_5 is "+ str(precision_5))
logging.info("ndcg_3 is "+ str(ndcg_3))
logging.info("ndcg_5 is "+ str(ndcg_5))

f1 = f1_score(y_test1, y_pred, average='micro')
logging.info("f1_score is "+ str(f1))
recall = recall_score(y_test1, y_pred, average='micro')
logging.info("recall_score is "+ str(recall))


