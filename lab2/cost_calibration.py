import math
import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_plan(operator, factors, weights):
    cost = 0.0
    for child in operator.children:
        cost += estimate_plan(child, factors, weights)

    if operator.is_hash_agg():
        # YOUR CODE HERE: hash_agg_cost = input_row_cnt * cpu_fac
        
        # 直接在这个操作算子下读取输入的行数
        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_hash_join():
        # hash_join_cost = (build_hashmap_cost + probe_and_pair_cost)
        #   = (build_row_cnt * cpu_fac) + (output_row_cnt * cpu_fac)
        # 为什么这个是output而不是input？这个是因为做Join操作要下推到孩子节点
        # 两个表做Join操作只需要构造一个hash table就行
        # 输出行的数量
        output_row_cnt = operator.est_row_counts()
        # 判断是左孩子做hash join还是右孩子做hash join
        build_side = int(operator.children[1].is_build_side())
        # 0表示左孩子，1表示右孩子，根据是哪个孩子来取孩子的行数量
        build_row_cnt = operator.children[build_side].est_row_counts()
        # 根据代价公式来算出估算的代价
        cost += (build_row_cnt + output_row_cnt) * factors['cpu']
        weights['cpu'] += (build_row_cnt + output_row_cnt)

    elif operator.is_sort():
        # YOUR CODE HERE: sort_cost = input_row_cnt * log(input_row_cnt) * cpu_fac

        input_row_cnt = operator.est_row_counts()
        if input_row_cnt != 0:
            cost += input_row_cnt * math.log2(input_row_cnt) * factors['cpu']
            weights['cpu'] += input_row_cnt * math.log2(input_row_cnt)
        

    elif operator.is_selection():
        # YOUR CODE HERE: selection_cost = input_row_cnt * cpu_fac

        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_projection():
        # YOUR CODE HERE: projection_cost = input_row_cnt * cpu_fac
        
        input_row_cnt = operator.est_row_counts()
        cost += input_row_cnt * factors['cpu']
        weights['cpu'] += input_row_cnt

    elif operator.is_table_reader():
        # YOUR CODE HERE: table_reader_cost = input_row_cnt * input_row_size * net_fac
        
        # input_row_cnt就是这个操作算子的评估行数
        # input_row_size可以通过操作算子的row_size()获得
        input_row_cnt = operator.est_row_counts()
        input_row_size = operator.row_size()
        cost += input_row_cnt * input_row_size * factors['net']
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_table_scan():
        # YOUR CODE HERE: table_scan_cost = row_cnt * row_size * scan_fac

        row_cnt = operator.est_row_counts()
        row_size = operator.row_size()
        cost += row_cnt * row_size * factors['scan']
        weights['scan'] += row_cnt * row_size

    elif operator.is_index_reader():
        # YOUR CODE HERE: index_reader_cost = input_row_cnt * input_row_size * net_fac
        
        input_row_cnt = operator.est_row_counts()
        input_row_size = operator.row_size()
        cost += input_row_cnt * input_row_size * factors['net']
        weights['net'] += input_row_cnt * input_row_size

    elif operator.is_index_scan():
        # YOUR CODE HERE: index_scan_cost = row_cnt * row_size * scan_fac
        
        row_cnt = operator.est_row_counts()
        row_size = operator.row_size()
        cost += row_cnt * row_size * factors['scan']
        weights['scan'] += row_cnt * row_size

    elif operator.is_index_lookup():
        # index_lookup_cost = net_cost + seek_cost
        #   = (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * net_fac +
        #     (build_row_cnt / batch_size) * seek_fac
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()
        build_row_size = operator.children[build_side].row_size()
        probe_row_cnt = operator.children[1 - build_side].est_row_counts()
        probe_row_size = operator.children[1 - build_side].row_size()
        batch_size = operator.batch_size()

        cost += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * factors['net']
        weights['net'] += (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size)

        cost += (build_row_cnt / batch_size) * factors['seek']
        weights['seek'] += (build_row_cnt / batch_size)

    else:
        print(operator.id)
        assert (1 == 2)  # unknown operator
    return cost


def estimate_calibration(train_plans, test_plans):
    # init factors
    factors = {
        "cpu": 1,
        "scan": 1,
        "net": 1,
        "seek": 1,
    }

    # get training data: factor weights and act_time
    est_costs_before = []
    act_times = []
    weights = []
    for p in train_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        # 得到粗略的总代价
        cost = estimate_plan(p.root, factors, w)
        # 把weights视作训练集,注意w是一个字典
        weights.append(w)
        # act_times看成是一个训练label
        act_times.append(p.exec_time_in_ms())
        # 粗略的代价
        est_costs_before.append(cost)

    # YOUR CODE HERE
    # calibrate your cost model with regression and get the best factors
    # factors * weights ==> act_time

    
    # 训练集是weights，转成矩阵类型的
    train_size = len(weights)
    # 生成 train_size * (len(factors))大小的矩阵
    X = np.zeros((train_size, len(factors)))
    for row_id in range(train_size):
        col_id = 0
        for value in weights[row_id].values():
            X[row_id][col_id] = value
            col_id += 1
    # act_times是一行，因此需要转置
    y = np.array(act_times).reshape(train_size, 1)
    # 假设满足线性关系：act_time = factors * weights + b 
    # 使用线性回归来求系数
    linreg = LinearRegression()
    #训练linreg,然后火区权重和截距
    linreg.fit(X, y)
    print('model intercept:', linreg.intercept_, 'model weights:', linreg.coef_)
    
    new_factors = {}
    i = 0
    for key in factors.keys():
        new_factors[key] = linreg.coef_[0][i]
        i += 1

    print("--->>> regression cost factors: ", new_factors)

    # evaluation
    est_costs = []
    for p in test_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, new_factors, w)
        est_costs.append(cost)

    return est_costs
