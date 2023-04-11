import torch
from torch import nn
import pandas as pd
import numpy as np


class querySentenceTransform:   # 查询语句特征化转换对象
    def __init__(self):
        pass

    def Word_Process(self, sentence):   # 实现将文本字符串型属性列数据转换为数值型
        symbols = ',.?。，()（）/*-+！!@#$￥%……^&-_ '
        word_2_idx = {}
        idx_2_word = {}
        i = 0
        for word in sentence:
            if word_2_idx.get(word) is None and word not in symbols:
                word_2_idx[word] = i
                idx_2_word[i] = word
                i += 1
        return word_2_idx, idx_2_word, i

    def data_num(self, date_time):  # 实现将日期的字符串型属性列数据转换为数值型
        datetime_new = int(date_time.replace('/', ''))
        return datetime_new

    def transform_bool(self, nums):
        new_nums = []
        length = len(nums)
        for idx in range(length):
            if nums[idx]:
                new_nums.append(1)
            elif not nums[idx]:
                new_nums.append(0)
        return new_nums


# 1）对查询语句进行特征转换
data = pd.read_csv("stu_table_tiny.csv")   # 读取查询语句内容
QST = querySentenceTransform()  # 声明一个对象

# 第一种情况，将文本型属性列转换为数值型
name_new = []
name = list(data['name'])
word2idx_name, idx2word_name, vocab_num_name = QST.Word_Process(sentence=name)
length1 = len(name)
for i in range(length1):
    name_new.append(word2idx_name.get(name[i]))

sex_new = []
sex = list(data['sex'])
word2idx_sex, idx2word_sex, vocab_num_sex = QST.Word_Process(sentence=sex)
length2 = len(sex)
for i in range(length2):
    sex_new.append(word2idx_sex.get(sex[i]))

education_new = []
education = list(data['education'])
word2idx_education, idx2word_education, vocab_num_education = QST.Word_Process(
    sentence=education)
length3 = len(education)
for i in range(length3):
    education_new.append(word2idx_education.get(education[i]))

# 第二种情况，将日期类型属性列转换为数值型
date_new = []
date = list(data['date_of_birth'])
length4 = len(date)
for i in range(length4):
    date_new.append(QST.data_num(date[i]))

enrollment_date_new = []
enrollment_date = list(data['enrollment_date'])
length5 = len(enrollment_date)
for i in range(length5):
    enrollment_date_new.append(QST.data_num(enrollment_date[i]))

# 第三种情况，将原本bool型属性列转换为数值型
is_married = list(data['is_married'])
is_married_new = QST.transform_bool(is_married)


# 第四种情况，将原本数值型属性列保留，为便于后续处理，将其转换为列表数据类型
age = list(data['age'])
age_new = list(np.array(data['age']))
grade = list(data['grade'])
grade_new = list(np.array(data['grade']))
average_score = list(data['average_score'])
average_score_new = list(np.array(data['average_score']))


table_new_columns = [name_new, sex_new, age_new, grade_new, date_new, average_score_new, education_new,
                     enrollment_date_new, is_married_new]

table_columns = [name, sex, age, grade, date,
                 average_score, education, enrollment_date, is_married]

table_columns_name = {'name': 0, 'sex': 1, 'age': 2, 'grade': 3, 'date_of_birth': 4, 'average_score': 5, 'education': 6,
                      'enrollment_date': 7}


class wordAnalyze:   # SQL查询语句一般的特征化向量转换
    def __init__(self):
        self.Tq = []
        self.Jq = [1]
        self.Cq = []
        self.table_num = 0
        self.query_sentence_num = 0

    def scanSql(self, sql_list):
        length = len(sql_list)
        vals = []
        for i in range(length):
            if sql_list[i] == 'FROM':
                i += 1
                while sql_list[i] != 'WHERE':
                    self.table_num += 1
                    i += 1
            elif sql_list[i] == 'WHERE':
                i += 1
                while i < length:
                    if sql_list[i] == 'AND':
                        i += 1
                    else:
                        idx = table_columns_name.get(sql_list[i])
                        for j in range(len(table_columns_name)):
                            if j == idx:
                                vals.append(1)
                            else:
                                vals.append(0)
                        min_val = min(np.array(table_new_columns[idx]))
                        max_val = max(np.array(table_new_columns[idx]))
                        real_val = table_columns[idx].count(
                            sql_list[i + 2].replace("'", ''))
                        val = str((real_val - min_val) / (max_val - min_val))
                        op = '1'
                        vals = ''.join(str(k) for k in vals)
                        temp = vals + op + val
                        self.Cq.append(temp)
                        self.query_sentence_num += 1
                        vals = []
                        i += 3
            else:
                i += 1
        array1 = [0]*self.table_num
        for idx in range(self.table_num):
            array1[idx] = 1
            temp = ''.join(str(k) for k in array1)
            self.Tq.append(temp)
            array1[idx] = 0

        return self.Tq, self.Jq, self.Cq, self.table_num, self.query_sentence_num


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 将该网络的训练移动至GPU上
print("Using device:", device)

# 1）首先将SQL语句根据空格将其进行拆分
sql_sentences_num = 4
sql_sentence1 = "SELECT is_married FROM student WHERE sex = 'female' AND education = 'doctor'"
sql_sentence2 = "SELECT is_married FROM student WHERE sex = 'male' AND education = 'bachelor'"
sql_sentence3 = "SELECT is_married FROM student WHERE sex = 'male' AND education = 'doctor'"
sql_sentence4 = "SELECT is_married FROM student WHERE sex = 'female' AND education = 'master'"
sql_array1 = sql_sentence1.split()
sql_array2 = sql_sentence2.split()
sql_array3 = sql_sentence3.split()
sql_array4 = sql_sentence4.split()

# 2）对查询语句特征化，转换为向量形式
word_analyze1 = wordAnalyze()
word_analyze2 = wordAnalyze()
word_analyze3 = wordAnalyze()
word_analyze4 = wordAnalyze()

Tq1, Jq1, Cq1, table_num1, query_sentence_num1 = word_analyze1.scanSql(
    sql_array1)
Tq2, Jq2, Cq2, table_num2, query_sentence_num2 = word_analyze2.scanSql(
    sql_array2)
Tq3, Jq3, Cq3, table_num3, query_sentence_num3 = word_analyze3.scanSql(
    sql_array3)
Tq4, Jq4, Cq4, table_num4, query_sentence_num4 = word_analyze4.scanSql(
    sql_array4)

# 3）计算查询语句实际的归一化后的基数估计值
a1, a2, a3, a4 = 0, 0, 0, 0
for idx in range(0, len(sex_new)):
    if sex[idx] == 'female' and education[idx] == 'doctor':
        a1 += 1
    elif sex[idx] == 'male' and education[idx] == 'bachelor':
        a2 += 1
    elif sex[idx] == 'male' and education[idx] == 'doctor':
        a3 += 1
    elif sex[idx] == 'female' and education[idx] == 'master':
        a4 += 1

a1 = a1 / len(name)
a2 = a2 / len(name)
a3 = a3 / len(name)
a4 = a4 / len(name)

real_cardinality = [a1, a2, a3, a4]


def stringToFloat(Tq, Jq, Cq):
    if len(Tq) >= 1:
        for i in range(len(Tq)):
            Tq[i] = float(Tq[i])

    if len(Jq) >= 1:
        for i in range(len(Jq)):
            Jq[i] = float(Jq[i])

    if len(Cq) >= 1:
        for i in range(len(Cq)):
            Cq[i] = float(Cq[i])

    length1 = len(Tq)
    length2 = len(Jq)
    length3 = len(Cq)
    if length1 <= 50:
        for i in range(length1, 50):
            Tq.append(0.5)

    if length2 <= 50:
        for i in range(length2, 50):
            Jq.append(0.5)

    if length3 <= 50:
        for i in range(length3, 50):
            Cq.append(0.5)

    Tq = [Tq]
    Cq = [Cq]
    Jq = [Jq]
    Tq = torch.tensor(Tq)
    Jq = torch.tensor(Jq)
    Cq = torch.tensor(Cq)

    Tq = torch.nn.functional.normalize(Tq, dim=1)
    Jq = torch.nn.functional.normalize(Jq, dim=1)
    Cq = torch.nn.functional.normalize(Cq, dim=1)

    return Tq, Jq, Cq

# 4）将其组合生成训练集

"""
# 采用特征化查询转换后生成的训练数据集
Tq1, Jq1, Cq1 = stringToFloat(Tq1, Jq1, Cq1)
Tq2, Jq2, Cq2 = stringToFloat(Tq2, Jq2, Cq2)
Tq3, Jq3, Cq3 = stringToFloat(Tq3, Jq3, Cq3)
Tq4, Jq4, Cq4 = stringToFloat(Tq4, Jq4, Cq4)

X1 = (Tq1, Jq1, Cq1)
X2 = (Tq2, Jq2, Cq2)
X3 = (Tq3, Jq3, Cq3)
X4 = (Tq4, Jq4, Cq4)
X = [X1, X2, X3, X4]
print(Tq1)
print(Cq1)
print(Jq1)

"""

# 此处为采用随机生成四组训练数据集
X1 = (
    torch.randn(10, 50, requires_grad=True),
    torch.randn(7, 50, requires_grad=True),
    torch.randn(30, 50, requires_grad=True)
)
X2 = (
    torch.randn(10, 50, requires_grad=True),
    torch.randn(7, 50, requires_grad=True),
    torch.randn(30, 50, requires_grad=True)
)
X3 = (
    torch.randn(10, 50, requires_grad=True),
    torch.randn(7, 50, requires_grad=True),
    torch.randn(30, 50, requires_grad=True)
)
X4 = (
    torch.randn(10, 50, requires_grad=True),
    torch.randn(7, 50, requires_grad=True),
    torch.randn(30, 50, requires_grad=True)
)
X = [X1, X2, X3, X4]


# 6）构造MSCNN多集合卷积神经网络模型


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 5) 随机初始化模型权重参数
        # 随机初始化权重
        self.input_size1 = 50
        self.hidden_size1 = 512
        self.output_size1 = 50

        self.input_size2 = 50
        self.hidden_size2 = 512
        self.output_size2 = 50

        self.input_size3 = 50
        self.hidden_size3 = 512
        self.output_size3 = 50

        self.input_size4 = 150
        self.hidden_size4 = 1024
        self.output_size4 = 1

        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(self.input_size1, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.output_size1),
            nn.ReLU(),

        )

        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(self.input_size2, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, self.output_size2),
            nn.ReLU(),

        )

        self.linear_relu_stack3 = nn.Sequential(
            nn.Linear(self.input_size3, self.hidden_size3),
            nn.ReLU(),
            nn.Linear(self.hidden_size3, self.output_size3),
            nn.ReLU(),
        )

        self.linear_relu_stack4 = nn.Sequential(
            nn.Linear(self.input_size4, self.hidden_size4),
            nn.ReLU(),
            nn.Linear(self.hidden_size4, self.output_size4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        Tq = x[0]
        Jq = x[1]
        Cq = x[2]
        logits1 = self.linear_relu_stack1(Tq)
        logits2 = self.linear_relu_stack2(Jq)
        logits3 = self.linear_relu_stack3(Cq)
        # print(logits1.shape)
        logits1 = torch.mean(logits1, axis=0)  # 列求平均
        logits2 = torch.mean(logits2, axis=0)
        logits3 = torch.mean(logits3, axis=0)
        # print(logits1)
        # print(logits1.shape)
        # 行进行拼接
        input_num = torch.cat([logits1, logits2, logits3])
        # print(input_num)
        # print(input_num.shape)
        logits4 = self.linear_relu_stack4(input_num)
        # print(logits4.shape)
        return logits4


a1 = torch.tensor([a1])
a2 = torch.tensor([a2])
a3 = torch.tensor([a3])
a4 = torch.tensor([a4])
y = (a1, a2, a3, a4)
learning_rate = 0.001

# 自定义神经网络的损失函数
def loss_fn(y_pred, y):
    # return (y - y_pred) ** 2
    return max(y_pred, y) / min(y_pred, y)


# loss_fn = nn.MSELoss()
model = NeuralNetwork()   # 创建神经网络对应的对象
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 神经网络参数优化器

for epoch in range(600):
    loss = torch.zeros(1)
    for i in range(len(X)):
        y_pred = model(X[i])
        # print(y_pred)
        loss = loss + loss_fn(y_pred, y[i])
        # print(y[i])
        # print(loss_fn(y_pred, y[i]))
    loss = loss / len(X)
    print(f"Mean q-loss = {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("测试训练集第一组：预测值 && 实际值")
print(model(X[0]))
print(y[0])
print("测试训练集第二组：预测值 && 实际值")
print(model(X[1]))
print(y[1])
print("测试训练集第三组：预测值 && 实际值")
print(model(X[2]))
print(y[2])
print("测试训练集第四组：预测值 && 实际值")
print(model(X[3]))
print(y[3])
