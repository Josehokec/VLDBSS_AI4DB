import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
import torch
import torch.nn as nn
import statistics as stats
import xgboost as xgb
# 导入pickle去保存加载模型
import pickle

def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v-min_v)/(max_v-min_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    feature = [] 
    # YOUR CODE HERE: extract features from query

    # table_stats.columns is dictionary 
    # range_query is an ParsedRangeQuery object
    # 遍历所有的列
    for i in range(len(considered_cols)):
        cur_col_name = considered_cols[i]
        min_val = table_stats.columns[cur_col_name].min_val()
        max_val = table_stats.columns[cur_col_name].max_val()
        # obtain sql statement cur col min value and max value 
        sql_min_val, sql_max_val = range_query.column_range(cur_col_name, min_val, max_val)
        # 对参数正则化,否则某个维度特征向量很大的话会造成效果不好
        l_normalized = (sql_min_val - min_val) / (max_val - min_val)
        r_normalized = (sql_max_val - min_val) / (max_val - min_val)
        feature.append(l_normalized)
        feature.append(r_normalized)
    
    # 把 AVI_sel, EBO_sel, Min_sel 三个值加到特征向量中
    AVI_sel = stats.AVIEstimator.estimate(range_query, table_stats)
    EBO_sel = stats.ExpBackoffEstimator.estimate(range_query, table_stats)
    Min_sel = stats.MinSelEstimator.estimate(range_query, table_stats)
    feature.append(AVI_sel)
    feature.append(EBO_sel)
    feature.append(Min_sel)
    return feature


def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        feature, label = None, None
        # YOUR CODE HERE: transform (query, act_rows) to (feature, label)
        # Some functions like rq.ParsedRangeQuery.parse_range_query and extract_features_from_query may be helpful.

        range_query = rq.ParsedRangeQuery.parse_range_query(query)
        # columns is all col name
        feature = extract_features_from_query(range_query, table_stats, columns)
        label = act_rows
        features.append(feature)
        labels.append(label)
    features = torch.tensor(features, dtype=torch.float32)
    # 这里获得的标签值最好取对数,而不是真实的行数量
    # 因为到时候预测出会出现负数,而预测的行数量只能大于等于0
    # 不做log2处理的话很难让模型训练出来
    # 做了log2处理之后模型预测的值是预测取了个对数,即模型预测的值是y,则估计的行数是2**y
    labels = torch.tensor(np.log2(labels), dtype=torch.float32)
    # 原始的labels是一行数据,现在将labels变成一列数据
    labels = labels.reshape(len(labels), 1)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip(*preprocess_queries(queries, table_stats, columns)))

    def __getitem__(self, index):
        return self.query_data[index]

    def __len__(self):
        return len(self.query_data)


def est_mlp(train_data, test_data, table_stats, columns):
    """
    est_mlp uses MLP to produce estimated rows for train_data and test_data
    """
    train_dataset = QueryDataset(train_data, table_stats, columns)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)
    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure
    
    # 使用nn.Sequential定义一个MLP
    model = nn.Sequential(
        nn.Linear(len(columns) * 2 + 3, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ) 

    # 初始化MLP的参数，这里采用遍历model每一层然后赋值
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # 这里采用高斯分布随机初始化，即参数w~N(0,2,1)
            nn.init.normal_(m.weight, 0, 1)

    # 定义损失函数和优化器，其中均方误差有现成的可以调用
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    num_epoch = 20
    print("train num is : ", num_epoch)
    for epoch in range(num_epoch):
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            optimizer.zero_grad()  
            # 注意: pred_Y现在是tensor, pred_Y预测的行数，而是行数的log2
            pred_Y =model(batch_X)
            # 根据预测的值和实际的值去计算损失                 
            cur_loss = criterion(batch_Y, pred_Y)
            # 将损失反向传播
            cur_loss.backward()
            # 更新梯度
            optimizer.step()
            # 每隔200批，看一次损失
            if i % 200 == 0:
                print("cur epoch: ", epoch, " | cur i: ", i, " | loss: ", cur_loss)

    # 将整个训练集丢进去看效果怎么样
    for i, data in enumerate(train_loader):
        (train_batch_X, train_batch_Y) = data
        pred_Y = model(train_batch_X)
        # 首先将预测的值转换成实际对行数的估计
        # 然后将tensor类型转换成1维list，然后用+把两个list拼接在一起
        train_est_rows += torch.exp2(pred_Y).reshape(-1).tolist()
        train_act_rows += torch.exp2(train_batch_Y).reshape(-1).tolist()

    test_dataset = QueryDataset(test_data, table_stats, columns)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure

    for (test_batch_X, test_batch_Y) in test_loader:
        # 使用模型去预测结果
        test_pred_Y = model(test_batch_X)
        # 然后将这一批数据拼接到最后的test列表中
        test_est_rows += torch.exp2(test_pred_Y).reshape(-1).tolist()
        test_act_rows += torch.exp2(test_batch_Y).reshape(-1).tolist()
    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def est_xgb(train_data, test_data, table_stats, columns):
    """
    est_xgb uses xgboost to produce estimated rows for train_data and test_data
    """
    # 直接根据preprocess_queries，拿到训练数据和训练标签
    # 注意标签是估计的行数取对数
    tensor_train_x, tensor_train_y = preprocess_queries(train_data, table_stats, columns)
    train_X = tensor_train_x.numpy()
    train_Y = tensor_train_y.numpy()

    train_est_rows, train_act_rows = [], []
    # YOUR CODE HERE: train procedure

    dtrain = xgb.DMatrix(train_X, label=train_Y)
    
    params = {
        'max_depth': 3,                     # 树最大深度，默认是10
        'eta': 0.5,                         # 学习率
        'reg_alpha': 0.01,                  # L1正则化系数
        'reg_lambda': 0.01,                 # L2正则化系数
        'objective': 'reg:squarederror'     # 损失函数是均方误差     
    } 
    """
    xgb.train的参数
        num_boost_round等价于n_estimators
        params：定义的参数列表
        dtrain：读取的训练数据
        num_boost_round：迭代次数
    """
    # 把数据丢进去训练,迭代20次
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=20)
    pred_Y = bst.predict(dtrain)
    # 需要改变形状
    train_act_rows = np.exp2(train_Y).reshape(-1).tolist()
    # 估计出行数量
    train_est_rows = np.exp2(pred_Y).tolist()

    # 测试集数据测试
    # 注意预测的值是对行的估计的对数
    tensor_test_x, tensor_test_y = preprocess_queries(test_data, table_stats, columns)
    test_est_rows, test_act_rows = [], []
    # YOUR CODE HERE: test procedure
    
    test_X = tensor_test_x.numpy()
    test_Y = tensor_test_y.numpy()
    dtest = xgb.DMatrix(test_X, label=test_Y)
    test_pred_Y = bst.predict(dtest)
    # 使用xgboost估算出行的数量
    test_est_rows = np.exp2(test_pred_Y).tolist()
    test_act_rows = np.exp2(test_Y).reshape(-1).tolist()
    # 需要保存xgboost model
    pickle.dump(bst, open("pima.pickle.dat", "wb"))
    return train_est_rows, train_act_rows, test_est_rows, test_act_rows

def get_xgb_model():
    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
    return loaded_model


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'mlp':
        est_fn = est_mlp
    else:
        est_fn = est_xgb

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)
    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = './data/title_stats.json'
    train_json_file = './data/query_train_20000.json'
    test_json_file = './data/query_test_5000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('mlp', train_data, test_data, table_stats, columns)
    # eval_model('xgb', train_data, test_data, table_stats, columns)
