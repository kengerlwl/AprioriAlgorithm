import numpy as np
import pandas as pd
import json
import os

f = open(os.path.abspath(os.path.dirname(__file__))+"/para.json", 'r')
para = json.loads(f.read())



def loadTest():
    data = np.loadtxt(open(os.path.abspath(os.path.dirname(__file__)) +'/../testData', "r"), dtype=str, delimiter=",", skiprows=1)
    data = data[0:para['testDataSize']]
    row, col = data.shape
    ans = []
    for i in range(row):
        tmp = []
        for j in range(col):
            if data[i][j]:
                tmp.append(data[i][j])
        ans.append(tmp)
    ans = np.array(ans)
    return ans

def loadDataSet():
    '''创建一个用于测试的简单的数据集'''

    data = pd.read_csv(open(os.path.abspath(os.path.dirname(__file__)) +'/../StudentsPerformance.csv'))
    data = data.sample(1000)

    print(data.shape)
    def gerDoodAndBad(score):
        ans = []

        # math judge
        if score['math score'] > 80:
            ans.append('MG')
        elif score['math score'] < 60:
            ans.append('MB')

        # reading judge
        if score['reading score'] > 85:
            ans.append('RG')
        elif score['reading score'] < 50:
            ans.append('RB')

        # writing judge
        if score['writing score'] > 85:
            ans.append('WG')
        elif score['writing score'] < 50:
            ans.append('WB')

        if score['test preparation course'] == 'completed':

            ans.append("preparation")
        ans.append(score['gender'])
        ans.append(score['race/ethnicity'])
        return ans

    data['Apriori'] = data.apply(lambda x: gerDoodAndBad(x), axis=1)
#     print(data.head())

    return np.array(data['Apriori'])
