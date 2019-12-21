import numpy as np
import scipy.sparse as sp
import torch


id2label = dict()


def encode_onehot(labels):
    classes = sorted(set(labels))

    classes_list = list(classes)
    for i in range(len(classes_list)):
        id2label[i] = classes_list[i]

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(data_type: str):
    if data_type == "cora":
        path, dataset = "data/cora/", "cora"
    elif data_type == "elliptic":
        path, dataset = "data/elliptic/", "elliptic"
    else:
        raise NotImplementedError("Data type not support!")

    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # map node name to id
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if data_type == "cora":
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        """
        # TODO: test code
        idx_train = []
        filter_labels = labels.numpy()
        for i in range(140):
            if filter_labels[i] != 0:
                idx_train.append(i)
        """
    elif data_type == "elliptic":
        filter_labels = labels.numpy()
        idx_train = [i for i in range(136265)if filter_labels[i] != 2]
        idx_val = [i for i in range(136000, 136275) if filter_labels[i] != 2]
        idx_test = [i for i in range(136265, 203769) if filter_labels[i] != 2]
    else:
        idx_train, idx_val, idx_test = 0, 0, 0

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def f1_score(output, labels):
    print("\n---------- Metrics for each label: ----------")
    for tid in id2label.keys():
        positive_label_id = tid
        preds = output.max(1)[1].type_as(labels)
        preds_num = preds.numpy().tolist()
        labels_num = labels.numpy().tolist()

        tp, fp, fn = 0, 0, 0
        for i in range(len(preds_num)):
            p, l = preds_num[i], labels_num[i]
            if p == positive_label_id:
                if l == p:
                    tp += 1
                else:
                    fp += 1
            elif l == positive_label_id:
                fn += 1
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        print(id2label[positive_label_id], precision, recall, f1)
    return None


if __name__ == '__main__':
    pass
