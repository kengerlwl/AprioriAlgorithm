import pandas as pd
import numpy as np


data = pd.read_csv("train.csv")
data =data.sample(100)
def gerDoodAndBad(score):
    ans = []

    # math judge
    if score['math score'] > 85:
        ans.append('MG')
    elif score['math score'] < 50:
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


    if score['test preparation course'] ==1:
        ans.append("preparation")

    return ans


data['grades'] = data.apply(lambda x: gerDoodAndBad(x), axis = 1 )

print(data.head(10))