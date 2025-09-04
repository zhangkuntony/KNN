import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def knn():
    # 1. 加载数据
    data_raw = load_iris()

    # 1.1. 获取特征，标签数据 + 特征工程（标准化）
    # 标准化数据
    # 特征
    print('data_raw.data: ', data_raw.data)
    data_feature = StandardScaler().fit_transform(data_raw.data)
    print('data_feature: ', data_feature)
    # 标签
    data_label = data_raw.target

    print('data_feature[:5]: ', data_feature[:5])
    print('data_label: ', data_label)
    print('data_raw.target_names: ', data_raw.target_names)

    # 2. 训练集、测试集分割
    # 验证集占数据集的20%
    x_train, x_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.2)

    # 3. 训练模型+交叉验证+网格搜索
    # 3.1. 实例化模型对象以及网格搜索
    estimator = KNeighborsClassifier()
    # 需要搜索的参数：
    parameter = {"n_neighbors": [5, 10, 15]}
    # 3折交叉验证+网格搜索
    estimator = GridSearchCV(estimator, parameter, cv=3)

    # 3.2. 使用网格搜索得到的参数进行训练
    estimator.fit(x_train, y_train)

    # 4. 预测
    result = estimator.predict(x_test)
    compare = [i==j for i, j in zip(result, y_test)]
    print(result)
    print(y_test)
    print(compare)

    print(estimator.best_params_)
    score = estimator.score(x_test, y_test)
    print(score)

if __name__ == '__main__':
    knn()
