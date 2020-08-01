# encoding=utf-8
from src.knrm import KNRM
import xgboost as xgb
import pickle


class KNRM_XGBOOST:

    def __init__(self, w2v):
        self.knrm = KNRM(w2v)

    def trian(self, train_path, save_model):
        X_train, y_train = self.knrm.get_features(train_path)
        model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True,
                                  objective='binary:logistic')
        model.fit(X_train, y_train)
        pickle.dump(model, open(save_model, "wb"))

    def test(self, test_path, model):
        X_test, y_test = self.knrm.get_features(test_path)
        ans = model.predict(X_test)
        # 计算准确率
        cnt1 = 0
        cnt2 = 0
        for i in range(len(y_test)):
            if ans[i] == y_test[i]:
                cnt1 += 1
            else:
                cnt2 += 1
        print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))