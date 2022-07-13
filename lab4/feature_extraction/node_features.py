import json

from feature_extraction.predicate_features import *


# change_alias2table changes alias to the real table name
"""
假如实际表名字是tablename1，改成别名为t1
alias2table应该是一个字典保存的是<实际表名字,别名表名字>
假设传入的参数：t1.columnname1 & alias2table
返回的是：tablename1.columnname1
"""
def change_alias2table(column, alias2table):
    relation_name = column.split('.')[0]
    column_name = column.split('.')[1]
    if relation_name in alias2table:
        return alias2table[relation_name] + '.' + column_name
    else:
        return column


# extract_feature_from_node extracts features from this node.
def extract_feature_from_node(node, alias2table):
    """
    主要功能是把节点转换成一个操作算子对象
    比如节点类型是Hash，而Hash对象仅仅存储了self.node_type变量且赋值了
    因此直接返回Hash()对象
    """
    relation_name, index_name = None, None
    if 'Relation Name' in node:
        relation_name = node['Relation Name']
    if 'Index Name' in node:
        index_name = node['Index Name']

    if node['Node Type'] == 'Sort':
        # YOUR CODE HERE: extract features from this Sort node.
        """
        这个部分要获取sort_keys,参考Aggregate是如何操作的
        如何知道节点的sort keys是啥样子的呢
        在encode_plan.py搜索for key in node['sort_keys']
        这些key可能有别名，因此需要统一起来
        """
        sort_keys = []
        if 'sort_keys' in node:
            
            sort_keys = [change_alias2table(key, alias2table) for key in node['sort_keys']]
        return Sort(sort_keys), None
    elif node['Node Type'] == 'Hash Join':
        return Join('Hash Join', pre2seq(node["Hash Cond"], alias2table, relation_name, index_name)), None
    elif node['Node Type'] == 'Hash':
        return Hash(), None
    elif node['Node Type'] == 'Merge Join':
        return Join('Merge Join', pre2seq(node["Merge Cond"], alias2table, relation_name, index_name)), None
    elif node['Node Type'] == 'Nested Loop':
        if 'Join Filter' in node:
            condition = pre2seq(node['Join Filter'], alias2table, relation_name, index_name)
        else:
            condition = []
        return Join('Nested Loop', condition), None
    elif node['Node Type'] == 'Aggregate':
        # 代码解释：聚集操作可能不止用一列来聚集这些key
        # 因此要获取所有聚类的key名字
        keys = []
        if 'Group Key' in node:
            keys = [change_alias2table(key, alias2table) for key in node['Group Key']]
        return Aggregate(node['Strategy'], keys), None
    elif node['Node Type'] == 'Seq Scan':
        # YOUR CODE HERE: extract features from this Seq Scan node.
        """
        需要返回Scan对象，具体需要设置以下变量
        node_type, condition_seq_filter, condition_seq_index, relation_name, index_name
        如何设置这些参数需要参考Index Scan,这里详细解释获取每个参数
        Seq Scan没有Index条件是不需要condition_seq_index
        plan的Seq Scan例子
        {
            "Plan Width": 4, "Actual Startup Time": 0.142, "Plan Rows": 1523, "Total Cost": 383558.84, 
            "Parent Relationship": "Outer", "Node Type": "Seq Scan","Relation Name": "movie_info", 
            "Filter": "((info)::text = ANY ('{Kyrgyzstan,Tunisia,Mozambique,Guyana}'::text[]))", 
            "Startup Cost": 0.0, "Rows Removed by Filter": 14835090, "Actual Rows": 630, "Alias": "mi", 
            "Actual Loops": 1, "Actual Total Time": 5475.27
        }
        """
        condition_seq_filter, condition_seq_index = [], []
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        relation_name, index_name = node["Relation Name"], node['Index Name']
        return Scan('Seq Scan', condition_seq_filter, None, relation_name, None), None
    elif node['Node Type'] == 'Index Scan':
        condition_seq_filter, condition_seq_index = [], []
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        relation_name, index_name = node["Relation Name"], node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) is not None:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name,
                        index_name), condition_seq_index
        else:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name), None
    else:
        raise Exception('Unsupported Node Type: ' + node['Node Type'])


class Aggregate(object):
    def __init__(self, strategy, keys):
        """
        按照什么strategy去聚这些keys
        比如一个表有<ID,username,department,grade>
        然后按照部门和年级去划分，此时keys就是department,grade>
        """
        self.node_type = 'Aggregate'
        self.strategy = strategy
        self.group_keys = keys

    def __str__(self):
        return 'Aggregate ON: ' + ','.join(self.group_keys)


class Sort(object):
    def __init__(self, sort_keys):
        self.sort_keys = sort_keys
        self.node_type = 'Sort'

    def __str__(self):
        return 'Sort by: ' + ','.join(self.sort_keys)


class Join(object):
    def __init__(self, node_type, condition_seq):
        self.node_type = node_type
        self.condition = condition_seq

    def __str__(self):
        return self.node_type + ' ON ' + ','.join([str(i) for i in self.condition])


class Scan(object):
    def __init__(self, node_type, condition_seq_filter, condition_seq_index, relation_name, index_name):
        self.node_type = node_type
        self.condition_filter = condition_seq_filter
        self.condition_index = condition_seq_index
        self.relation_name = relation_name
        self.index_name = index_name

    def __str__(self):
        return self.node_type + ' ON ' + ','.join([str(i) for i in self.condition_filter]) + '; ' + ','.join(
            [str(i) for i in self.condition_index])


class Hash(object):
    def __init__(self):
        self.node_type = 'Hash'

    def __str__(self):
        return 'Hash'


def class2json(instance):
    if instance is None:
        return json.dumps({})
    else:
        return json.dumps(todict(instance))


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
                     for key, value in obj.__dict__.items()
                     if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj