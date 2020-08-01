# encoding=utf-8
from src.knrm_xgboost import KNRM_XGBOOST
from gensim.models import KeyedVectors
import xgboost as xgb


w2v = KeyedVectors.load('./model/w2v/w2v.model')
kx = KNRM_XGBOOST(w2v)
kx.trian('./data/train.csv', './model/knrm_xgboost.model')

# model = xgb.Booster(model_file='./model/knrm_xgboost.model')
# kx.test('./data/test.csv', model)