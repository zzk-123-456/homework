import utils
from utils import load_data, plot_error, up_sample, pca
from sk_models import *
from nn_models import *
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True  # 确保每次结果相同
torch.backends.cudnn.benchmark = False  # 关闭 cudnn 的自动调优


def train_with_sk():
    X_train, X_test, Y_train, Y_test = load_data()
    X_train, Y_train = up_sample(X_train, Y_train)
    # X_train, X_test = pca(X_train, X_test, 10)
    param_svm = {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 1]}
    param_rf = {'max_depth': [10, 15, 20, 25, 30], 'min_samples_leaf': [1, 5, 10]}
    param_knn = {'n_neighbors': np.arange(1, 20)}
    best_param_svm, best_param_rf, best_param_knn = None, None, None
    best_param_svm = svm_class(X_train, X_test, Y_train, Y_test, param_svm)
    best_param_rf = randomforest_class(X_train, X_test, Y_train, Y_test, param_rf)
    best_param_knn = knn_class(X_train, X_test, Y_train, Y_test, param_knn)
    return best_param_svm, best_param_rf, best_param_knn


def train_with_nn():
    X_train, X_test, Y_train, Y_test = load_data()
    model1 = MLPClassifier(input_dim=20, num_classes=4, lr=2e-3
                             , batch_size=32, train_epochs=20, hidden_size=32, device="cuda", model_mode='single')
    model2 = MLPClassifier(input_dim=20, num_classes=4, lr=2e-3
                             , batch_size=32, train_epochs=20, hidden_size=32, device="cuda", model_mode='double')
    model3 = MLPClassifier(input_dim=20, num_classes=4, lr=2e-3
                             , batch_size=32, train_epochs=20, hidden_size=32, device="cuda", model_mode='linear')
    # 捕捉输入维度不匹配异常
    try:
        train_loss1, valid_loss1 = model1.fit(torch.tensor(X_train), torch.tensor(Y_train))
        train_loss2, valid_loss2 = model2.fit(torch.tensor(X_train), torch.tensor(Y_train))
        train_loss3, valid_loss3 = model3.fit(torch.tensor(X_train), torch.tensor(Y_train))
        test_acc1 = model1.score(torch.tensor(X_test), torch.tensor(Y_test))
        print("model1 acc is %.4f" % test_acc1)
        test_acc2 = model2.score(torch.tensor(X_test), torch.tensor(Y_test))
        print("model2 acc is %.4f" % test_acc2)
        test_acc3 = model3.score(torch.tensor(X_test), torch.tensor(Y_test))
        print("model3 acc is %.4f" % test_acc3)
        plot_error(train_loss1, valid_loss1, train_loss2, valid_loss2, train_loss3, valid_loss3)
    except Exception as e:
        print(f"发生异常: {e}")
    return



if __name__ == '__main__':
    # train_with_sk()
    # train_with_nn()
    utils.cal_kl()