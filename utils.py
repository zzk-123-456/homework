import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data():
    path = r'C:\Users\zhuxf\手机价格分类'
    train_data = pd.read_csv(os.path.join(path, "train.csv"))
    data = train_data.iloc[:, :-1]
    label = train_data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test = X_train.to_numpy(), X_test.to_numpy(), Y_train.to_numpy(), Y_test.to_numpy()
    # 2. 初始化标准化器
    scaler = StandardScaler()

    # 3. 训练标准化器，计算训练集的均值和标准差
    scaler.fit(X_train)

    # 4. 用训练集的标准化器转换训练集
    X_train_scaled = scaler.transform(X_train)

    # 5. 用同一个标准化器对测试集进行标准化
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, Y_train, Y_test

# 上采样
def up_sample(data, label):
    ros = RandomOverSampler(random_state=42)
    train_data, train_label = ros.fit_resample(data, label)
    return train_data, train_label


# 下采样
def down_sample(data, label):
    rus = RandomUnderSampler(random_state=42)  # random_state为0（此数字没有特殊含义，可以换成其他数字），使得每次代码运行的结果保持一致
    train_data, train_label = rus.fit_resample(data, label)  # 使用原始数据的特征变量和目标变量生成欠采样数据集
    return train_data, train_label


def plot_error(train_loss_1, valid_loss_1, train_loss_2, valid_loss_2, train_loss_3, valid_loss_3):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 行 1 列，调整图形大小
    epochs = np.arange(len(train_loss_1))

    # 第一个子图：model1
    axs[0].plot(epochs, train_loss_1, color='g', label='train')
    axs[0].plot(epochs, valid_loss_1, color='r', label='valid')
    axs[0].set_title('model loss with single layer')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].legend()

    # 第二个子图：model2
    axs[1].plot(epochs, train_loss_2, color='g', label='train')
    axs[1].plot(epochs, valid_loss_2, color='r', label='valid')
    axs[1].set_title('model loss with double layer')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('loss')
    axs[1].legend()

    # 第三个子图：model3
    axs[2].plot(epochs, train_loss_3, color='g', label='train')
    axs[2].plot(epochs, valid_loss_3, color='r', label='valid')
    axs[2].set_title('model loss with linear layer')
    axs[2].set_xlabel('epochs')
    axs[2].set_ylabel('loss')
    axs[2].legend()

    # 调整布局以防止标签重叠
    plt.tight_layout()
    plt.savefig('nn_figs.png')

    # 显示图形
    plt.show()


def pca(X_train, X_test, dim=5):
    X = np.concatenate([X_train, X_test])
    pca_emb = PCA(n_components=dim)
    pca_emb.fit(X)
    X_train_trans = pca_emb.transform(X_train)
    X_test_trans = pca_emb.transform(X_test)
    return X_train_trans, X_test_trans


def cal_kl():
    X_train, X_test, Y_train, Y_test = load_data()
    X_1 = X_train[Y_train == 0]
    X_2 = X_train[Y_train == 1]
    X_3 = X_train[Y_train == 2]
    X_4 = X_train[Y_train == 3]

    hist1, bins1 = np.histogram(X_1, bins=30, density=True)
    hist2, bins2 = np.histogram(X_2, bins=30, density=True)
    hist3, bins3 = np.histogram(X_3, bins=30, density=True)
    hist4, bins4 = np.histogram(X_4, bins=30, density=True)

    hist = [hist1, hist2, hist3, hist4]
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            kl = entropy(hist[i] + 1e-10, hist[j] + 1e-10)
            print(f"{i},{j},{kl}")