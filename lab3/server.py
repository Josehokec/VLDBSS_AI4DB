from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import pickle
import sys
# sys.path.append('.\\')
sys.path.append(r"C:\Users\Austin LIU\vldbss2022-labs-Josehokec")
# 从lab1去加载range_query和statistics文件
import lab1.statistics as stats
import lab1.range_query as rq
# 不得已把learn from query拉到lab3中
import learn_from_query as lfq
import xgboost as xgb
# 加载lab2的py文件
import lab2.plan as norm_plan
import lab2.cost_calibration as cc
import ast

host = ('localhost', 8888)

stats_json_file = '.\\lab1/data/title_stats.json'
columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
# 从现有的数据表中读取数据
# 然后提取摘要存在table_stats中
table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)

lab1_model = pickle.load(open(r"C:\Users\Austin LIU\vldbss2022-labs-Josehokec\lab3\pima.pickle.dat", "rb"))

class Resquest(BaseHTTPRequestHandler):
    def handle_cardinality_estimate(self, req_data):
        # YOUR CODE HERE: use your model in lab1
        """
        补全的代码逻辑：
        req_data拿到的数据是字节类型的，需要转成str
        样子是：b'kind_id>1 and kind_id<10'
        为了调用ParsedRangeQuery，需要将语句补全
        select * from imdb.title where
        补全的sql语句调用parse_range_query转成特征向量
        preprocess_queries
        然后放在模型中去做预测
        Notice: 没有做异常处理
        """
        # print("cardinality_estimate post_data: " + str(req_data))
        sql_query = "select * from imdb.title where " + str(req_data)[2:-1]
        # print("sql_statement: ", sql_query)
        range_query = rq.ParsedRangeQuery.parse_range_query(sql_query)
        # columns is all col name
        feature = lfq.extract_features_from_query(range_query, table_stats, columns)
        # print("feature: ", feature)
        data_test = xgb.DMatrix([feature], [0])
        # 注意之前预测的是行数取了对数
        pred_rows_log2 = lab1_model.predict(data_test)
        sel = 2 ** pred_rows_log2[0] / table_stats.row_count
        # print("selectivity: ", sel)
        return {"selectivity": sel, "err_msg": ""} # return the selectivity

    def handle_cost_estimate(self, req_data):
        # YOUR CODE HERE: use your model in lab2
        # print("cost_estimate post_data: " + str(req_data))
        """
        req_data是bytes类型的，可以通过decode()转成字符串
        
        handle_cost_estimate函数处理步骤：
        1.将mysql_plan转成json格式
        2.取plan翻译成plan，返回的是plan的Operation的根节点而不是plan
        3.生成一个Plan对象
        4.加载lab2得到的回归参数
        5.使用estimate_plan(operator, factors, weights)计算cost
        """
        mysql_plan = req_data.decode()
        # print("\nserver --> mysql_plan:", mysql_plan)
        mysql_plan_json = json.loads(mysql_plan)
        
        root_op = norm_plan.Plan.parse_mysql_plan(mysql_plan_json)
        cur_plan = norm_plan.Plan(None, root_op)
        # w保存的是各个算子操作的数量是多少，即权重
        regression_factors = {
            "cpu": 1.981e-05, 
            "scan": 1.521e-05, 
            "net": -2.619e-06, 
            "seek": 171.512
        }

        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        regression_cost = cc.estimate_plan(cur_plan.root, regression_factors, w)
        print("regression_cost", regression_cost)
        
        # regression_cost = 10.0
        return {"cost": regression_cost, "err_msg": ""} # return the cost

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        req_data = self.rfile.read(content_length)
        resp_data = ""
        if self.path == "/cardinality":
            resp_data = self.handle_cardinality_estimate(req_data)
        elif self.path == "/cost":
            resp_data = self.handle_cost_estimate(req_data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
