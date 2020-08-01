# encoding=utf-8
import jieba
import numpy as np


def kernel_mu(n_kernels, manual=False):
    if manual:
        return [1, 0.95, 0.90, 0.85, 0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.80, -0.85, -0.90, -0.95]
    mus = [1]
    if n_kernels == 1:
        return mus
    bin_step = (1-(-1))/(n_kernels-1)
    mus.append(1-bin_step/2)
    for k in range(1, n_kernels-1):
        mus.append(mus[k]-bin_step)
    return mus


def kernel_sigma(n_kernels):
    sigmas = [0.001]
    if n_kernels == 1:
        return sigmas
    return sigmas+[0.1]*(n_kernels-1)


def remask(mask, n_gram=1):
    return mask[:, n_gram-1:]


def gen_mask(q, d):
    mask = np.zeros((len(q), len(q[0]), len(d[0])))
    for b in range(len(q)):
        mask[b, :np.count_nonzero(q[b]), :np.count_nonzero(d[b])] = 1
    return mask


def pooling_emb(vector, n_gram=1):
    if n_gram == 1:
        return vector
    vec = np.zeros((vector.shape[0], vector.shape[1]-(n_gram-1), vector.shape[2]))
    for index in range(vector.shape[1]-n_gram+1):
        vec[:, index] = np.mean(vector[:, index:index+n_gram],1)
    return vec


def normalize(seq_emb):
    assert isinstance(seq_emb, np.ndarray)
    assert seq_emb.ndim in (2, 3)
    return norm(seq_emb, axis=seq_emb.ndim-1)


def norm(X, axis):
    return np.nan_to_num(X/np.linalg.norm(X, 2, axis=axis, keepdims=True), copy=False)


class KNRM:

    def __init__(self, w2v, max_len=20):
        self.w2v = w2v
        self.max_len = max_len
        self.mus = np.array(kernel_mu(11))[np.newaxis, np.newaxis, np.newaxis, :]
        self.sigmas = np.array(kernel_sigma(11))[np.newaxis, np.newaxis, np.newaxis, :]

    def build(self, q_emb, d_emb):
        mm_total = []
        for q_n in range(3):
            for d_n in range(3):
                qt_mask = gen_mask(remask(q_emb, n_gram=q_n+1), remask(d_emb, n_gram=d_n+1))[:, :, :, np.newaxis]
                q_emb_norm = normalize(pooling_emb(q_emb, n_gram=q_n+1))
                d_emb_norm = normalize(pooling_emb(d_emb, n_gram=d_n+1))
                log_pooling_sum = self.interaction_matrix(q_emb_norm, d_emb_norm, qt_mask)
                mm_total.append(log_pooling_sum)
        pooling_sum = np.concatenate(mm_total, axis=1)
        return pooling_sum

    def interaction_matrix(self, q_emb_norm, d_emb_norm, qt_mask):
        match_matrix = np.matmul(q_emb_norm, np.transpose(d_emb_norm, (0, 2, 1)))[:, :, :, np.newaxis]
        kernel_pooling = np.exp(-((match_matrix-self.mus)**2)/(2*(self.sigmas**2)))
        kernel_pooling = kernel_pooling*qt_mask
        pooling_row_sum = np.sum(kernel_pooling, 2)
        log_pooling = np.log(np.clip(pooling_row_sum, a_min=1e-10, a_max=np.inf)) * .01
        log_pooling_sum = np.sum(log_pooling, 1)
        return log_pooling_sum

    def get_features(self, data_path):
        q_emb = []
        d_emb = []
        labels = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                _, query, doc, label = line.replace('\n', '').split('\t')
                q_seq = []
                d_seq = []
                for word in jieba.lcut(query.strip()):
                    if word in self.w2v:
                        q_seq.append(self.w2v[word])
                if len(q_seq) < self.max_len:
                    for _ in range(self.max_len-len(q_seq)):
                        q_seq.append(np.asarray([0] * 300))
                else:
                    q_seq = q_seq[:self.max_len]
                for word in jieba.lcut(doc.strip()):
                    if word in self.w2v:
                        d_seq.append(self.w2v[word])
                if len(d_seq) < self.max_len:
                    for _ in range(self.max_len-len(d_seq)):
                        d_seq.append(np.asarray([0] * 300))
                else:
                    d_seq = d_seq[:self.max_len]
                q_emb.append(q_seq)
                d_emb.append(d_seq)
                labels.append(int(label))
        q_emb = np.array(q_emb)
        d_emb = np.array(d_emb)
        labels = np.array(labels)
        return self.build(q_emb, d_emb), labels


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    w2v = KeyedVectors.load('../model/w2v/w2v.model')
    knrm = KNRM(w2v)
    m = knrm.get_features('../data/train.csv')