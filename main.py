# encoding=utf-8
from src.knrm_xgboost import KNRM_XGBOOST
from gensim.models import KeyedVectors
import pickle


w2v = KeyedVectors.load('./model/w2v/w2v.model')
kx = KNRM_XGBOOST(w2v)
kx.trian('./data/train.csv', './model/knrm_xgboost.model')

# model = pickle.load(open('./model/knrm_xgboost.model', 'rb'))
# kx.test('./data/test.csv', model)
# 82.34%