import cv2
import numpy as np
import os
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import MiniBatchKMeans

def siftExtract(data_path):
    """
    函数功能：读取图片并提取SIFT特征
    参数说明：
        data_path: 数据集的根路径
    返回值：训练数据、测试数据、训练标签、测试标签
    """
    sift = cv2.SIFT_create()
    sub_folder_list = os.listdir(data_path)
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for sub_folder in sub_folder_list:
        file_list = os.listdir(data_path + "/" + sub_folder)
        i = 1
        for file in file_list:
            img = cv2.imread(data_path + "/" + sub_folder + "/" + file)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            des = sift.detectAndCompute(img_grey, None)
            if i <= 150:
                train_data.append(des[1])
                train_label.append(sub_folder)
            else:
                test_data.append(des[1])
                test_label.append(sub_folder)
            i += 1

    return train_data, test_data, train_label, test_label

def kmeans(train_data, clusters_num):
    """
    函数功能：使用KMeans算法对SIFT特征进行聚类
    参数说明：
        train_data: 训练数据
        clusters_num: 聚类的簇数
    返回值：KMeans聚类模型
    """
    kmeans_data = np.vstack(train_data)
    kmeans = MiniBatchKMeans(n_clusters=clusters_num)
    kmeans.fit(kmeans_data)
    return kmeans

def sift2WordBagVector(data, kmeans):
    """
    函数功能：将SIFT特征转换为词袋模型
    参数说明：
        data: 数据集
        kmeans: KMeans聚类模型
    返回值：词袋模型
    """
    wordbag = []
    for d in data:
        class_list = kmeans.predict(d)
        hist, _ = np.histogram(class_list, bins=np.arange(kmeans.n_clusters + 1))
        wordbag.append(hist)
    return wordbag

# Tips: 没必要用GPU训练，因为数据量不大，而且SVM本身就是CPU训练的
def train(wordbag_path, label_path, save_model_path):
    """
    函数功能：使用svm模型训练的方法
    参数说明：
        wordbag_path: 训练集词袋模型的路径
        label_path: 训练集标签的路径
        save_model_path: 模型保存的路径
    返回值：无
    """
    train_data = np.load(wordbag_path).astype(np.float32)
    train_label = np.load(label_path).astype(np.int32)
    svm = SVC(kernel='rbf', C=1000, decision_function_shape='ovo')
    svm.fit(train_data, train_label)

    with open(save_model_path, 'wb') as f:
        pickle.dump(svm, f)


def test(data_path, model_path, test_data_path, test_label_path, conf_matrix_path):
    """
    函数功能：测试模型的方法
    参数说明：
        data_path: 数据集的根路径
        model_path: 模型的路径
        test_data_path: 测试集数据的路径
        test_label_path: 测试集标签的路径
        conf_matrix_path: 生成的混淆矩阵的保存路径
    返回值：无
    """
    test_data = np.load(test_data_path).astype(np.float32)
    test_label = np.load(test_label_path).astype(np.int32)
    with open(model_path, 'rb') as f:
        svm = pickle.load(f)
    pred_label = svm.predict(test_data)

    print('Results of different classes:')
    print(classification_report(test_label, pred_label))
    conf_matrix = confusion_matrix(test_label, pred_label)
    cnf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix_norm = np.around(cnf_matrix_norm, decimals=2)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cnf_matrix_norm, annot=True, cmap='Greens')
    class_list = os.listdir(data_path)
    plt.ylim(0, 15)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=90)
    plt.yticks(tick_marks, class_list, rotation=45)
    plt.savefig(conf_matrix_path)
    plt.show()

    
def main():
    data_path = "./data/15-Scene"
    clusters_num = [128, 256, 512]

    for cluster_num in clusters_num:
        print(f"Start to extract SIFT features and save the result with cn = {cluster_num}...")
        train_data, test_data, train_label, test_label = siftExtract(data_path)
        kmeans_model = kmeans(train_data, cluster_num)
        train_wordbag = sift2WordBagVector(train_data, kmeans_model)
        test_wordbag = sift2WordBagVector(test_data, kmeans_model)
        np.save(f"./data/train_wordbag_cn={cluster_num}.npy", np.array(train_wordbag))
        np.save(f"./data/train_label_cn={cluster_num}.npy", np.array(train_label))
        np.save(f"./data/test_wordbag_cn={cluster_num}.npy", np.array(test_wordbag))
        np.save(f"./data/test_label_cn={cluster_num}.npy", np.array(test_label))

    print("Result saved successfully!")

    for cluster_num in [128, 256, 512]:
        train(
            f'./data/train_wordbag_cn={cluster_num}.npy', 
            f'./data/train_label_cn={cluster_num}.npy', 
            f'./data/svm_cn={cluster_num}.pickle'
        )

    for cluster_num in [128, 256, 512]:
        test(
            './data/15-Scene',
            f'./data/svm_cn={cluster_num}.pickle',
            f'./data/test_wordbag_cn={cluster_num}.npy',
            f'./data/test_label_cn={cluster_num}.npy',
            f'./data/conf_matrix_cn={cluster_num}.png'
        )

main()
