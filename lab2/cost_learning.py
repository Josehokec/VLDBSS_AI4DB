import torch
import torch.nn as nn
import torch.nn.functional as F
from plan import Operator
import numpy as np
operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp"]


# There are many ways to extract features from plan:
# 1. The simplest way is to extract features from each node and sum them up. For example, we can get
#      a  the number of nodes;
#      a. the number of occurrences of each operator;
#      b. the sum of estRows for each operator.
#    However we lose the tree structure after extracting features.
# 2. The second way is to extract features from each node and concatenate them in the DFS traversal order.
#                  HashJoin_1
#                  /          \
#              IndexJoin_2   TableScan_6
#              /         \
#          IndexScan_3   IndexScan_4
#    For example, we can concatenate the node features of the above plan as follows:
#    [Feat(HashJoin_1)], [Feat(IndexJoin_2)], [Feat(IndexScan_3)], [Feat(IndexScan_4)], [Padding], [Feat(TableScan_6)], [Padding]
#    Notice1: When we traverse all the children in DFS, we insert [Padding] as the end of the children. In this way, we
#    have an one-on-one mapping between the plan tree and the DFS order sequence.
#    Notice2: Since the different plans have the different number of nodes, we need padding to make the lengths of the
#    features of different plans equal.
class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: define variables to collect features from plans
        self.features = []
        # 调试用的，输出特征
        # self.features_info = []

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: extract features from op
        # op_feature存储的是当前这个算子的特征
        # 如果这个算子是空的话，那么需要回到父亲节点，然后补一个Zero特征
        op_feature  = [0] * 11
        if op == None:
            self.features.append(op_feature) 
            # self.features_info.append("Zero")
            return 
        
        # self.features_info.append("Operator")
        if op.is_projection():
            op_feature[0] = 1
        elif op.is_selection():
            op_feature[1] = 1
        elif op.is_sort():
            op_feature[2] = 1
        elif op.is_hash_agg():
            op_feature[3] = 1
        elif op.is_hash_join():
            op_feature[4] = 1
        elif op.is_table_scan():
            op_feature[5] = 1
        elif op.is_index_scan():
            op_feature[6] = 1
        elif op.is_table_reader():
            op_feature[7] = 1
        elif op.is_index_reader():
            op_feature[8] = 1
        elif op.is_index_lookup():
            op_feature[9] = 1
        op_feature[10] = op.est_row_counts()
        # 把特征拼接起来
        self.features.append(op_feature)

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        # YOUR CODE HERE: process and return the features
        self.add_operator(None)
        # 遍历到叶子节点，则补上空算子,回退，实际上就是前序遍历算法
        # 可以通过self.features_info查看是实际算子还是补None
        return self.features


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operator_num):
        super().__init__()
        self.data = []
        for plan in plans:
            collector = PlanFeatureCollector()
            vec = collector.walk_operator_tree(plan.root)
            # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
            """
            vec形状如下：
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3300.0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3300.0],
             ... 
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3300.0]]
            """
            # max_operator_num等同于最大的特征向量长度
            # 比如sql有2个操作算子A->B，则实际特征是A B None None,此时max_operator_num=4
            # 每个操作算子含有11个特征:10(onehot) + 1 (est_rows) 
            
            assert (len(vec) <= max_operator_num)
            if len(vec) != max_operator_num:
                padding = [0] * 11
                for _ in range(len(vec), max_operator_num):
                    vec.append(padding)
            assert (len(vec) == max_operator_num)
            # 将特征反转，不然lstm到最后一直输出0
            # vec = list(reversed(vec))
            features = torch.Tensor(vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.data.append((features, exec_time))    

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define your model for cost estimation
class YourModel(nn.Module):
    def __init__(self, input_size=24*11):
        super().__init__()
        # YOUR CODE HERE
        """
        当batch_first=True时, x的形状为：(batch_size,seq_len,input_size)
        传入的X向量是一个三维的数据：第一维度是这批数据大小，第二维度是顺序步数，第三维度是是输入的大小
        比如当前实验每一批数据是10个，一共最多24个操作算子，一个操作算子11个特征
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers默认是1，当batch_first设置为True
        hidden_size:是自己随便设置的
        # lstm效果太差了
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.layer3 = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), 0.0001)
        """
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), 0.0001)

    def forward(self, x):
        #传进来的x是 batch_size, 24*11的torch
        batch_size = len(x)
        x = torch.reshape(x, (batch_size, 24*11))
        return self.model(x)

    def init_weights(self):
        # YOUR CODE HERE
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                # normal_是高斯分布随机初始化，即参数w ~ N(0, 1)
                nn.init.xavier_uniform_(m.weight)


def count_operator_num(op: Operator):
    """
    功能：输出实际的查询算子
    假设计划是：
        Projection_4
          --IndexReader_7
              --Selection_6
                  --IndexRangeScan_5
    则返回的是实际操作算子的两倍，因为要用空值补全
    """
    num = 2  # one for the node and another for the end of children
    for child in op.children:
        num += count_operator_num(child)
    return num


def check_operator(op: Operator):
    print(op.id)
    for child in op.children:
        check_operator(child)


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")
    train_dataset = PlanDataset(train_plans, max_operator_num)
    # 300 -> 16 批大小设置成16效果最好
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    model = YourModel()
    model.init_weights()

    def loss_fn(est_time, act_time):
        # YOUR CODE HERE: define loss function
        loss = torch.mean(torch.abs(act_time-est_time)/act_time)
        return loss

    # YOUR CODE HERE: complete training loop
    # 训练的轮数要大于250次才能看到好的效果
    num_epoch = 500
    print("start training...")
    for epoch in range(num_epoch):
        for step, (train_x,train_y) in enumerate(train_loader): 
            pred_y = model.forward(train_x)
            loss = loss_fn(pred_y, train_y)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            if step % 20 == 0 and epoch % 50 == 0:
                print("cur epoch is: ", epoch, "cur step is: ", step, "cur loss is: ", loss)

    train_est_times, train_act_times = [], []
    for _, (train_x,train_y) in enumerate(train_loader):
        # YOUR CODE HERE: evaluate on train data
        train_pred_y = model.forward(train_x)
        # 将其变成一行多列的数据然后转list
        train_est_times += train_pred_y.reshape(-1).tolist()
        train_act_times += train_y.reshape(-1).tolist()

    test_dataset = PlanDataset(test_plans, max_operator_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=0)

    test_est_times, test_act_times = [], []
    for _, (test_x,test_y) in enumerate(test_loader):
        # YOUR CODE HERE: evaluate on test data
        test_pred_y = model.forward(test_x)
        test_est_times += test_pred_y.reshape(-1).tolist()
        test_act_times += test_y.reshape(-1).tolist()
    """
    els = [round(x,2) for x in train_est_times]
    alt = [round(x,2) for x in train_act_times]
    print(els[20:40])
    print(alt[20:40])
    """
    return train_est_times, train_act_times, test_est_times, test_act_times
