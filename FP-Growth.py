import fp_growth_py3 as fpg
import numpy as np
import pandas as pd
import copy
# 数据集
def loadDataSet():
    '''创建一个用于测试的简单的数据集'''

    data = pd.read_csv("train.csv")
    data = data.sample(1000)

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

        if score['test preparation course'] == 1:
            ans.append("preparation")

        return ans

    data['Apriori'] = data.apply(lambda x: gerDoodAndBad(x), axis=1)

    return np.array(data['Apriori'])



def calcConf(freqSet, H, supportData, brl, minConf=0.7):  # 规则生成与评价
    '''
        计算规则的可信度，返回满足最小可信度的规则。
        freqSet(frozenset):频繁项集
        H(frozenset):频繁项集中所有的元素
        supportData(dic):频繁项集中所有元素的支持度
        brl(tuple):满足可信度条件的关联规则
        minConf(float):最小可信度
    '''
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH



def scanD(D, Ck, minSupport):
    '''
        计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
        返回满足最小支持度的项集的集合，和所有项集支持度信息的字典。
    '''
    ssCnt = {}
    for tid in D:  # 对于每一条transaction
        for can in Ck:  # 对于每一个候选项集can，检查是否是transaction的一部分 # 即该候选can是否得到transaction的支持
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    # print("ssCnt is",ssCnt)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems  # 每个项集的支持度
        if support >= minSupport:  # 将满足最小支持度的项集，加入retList
            retList.insert(0, key)
        supportData[key] = support  # 汇总支持度数据
    return retList, supportData


def aprioriGen(Lk, k):  # Aprior算法
    '''
        两相比较，如果前k-1个元素一样，那么第k个元素一定不一样，那么可以生成出k+1个元素大小的元素，时间复杂度为n^2 *(n log(n))

        由初始候选项集的集合Lk生成新的生成候选项集，
        k表示生成的新项集中所含有的元素个数
        注意其生成的过程中，首选对每个项集按元素排序，然后每次比较两个项集，只有在前k-1项相同时才将这两项合并。这样做是因为函数并非要两两合并各个集合，那样生成的集合并非都是k+1项的。在限制项数为k+1的前提下，只有在前k-1项相同、最后一项不相同的情况下合并才为所需要的新候选项集。
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[: k - 2];
            L2 = list(Lk[j])[: k - 2];
            L1.sort()
            L2.sort()
            # print(L1,L2)
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
        对频繁项集中元素超过2的项集进行合并。
        freqSet(frozenset):频繁项集
        H(frozenset):频繁项集中的所有元素，即可以出现在规则右部的元素
        supportData(dict):所有项集的支持度信息
        brl(tuple):生成的规则
    '''
    m = len(H[0])
    if len(freqSet) > m + 1:  # 查看频繁项集是否足够大，以到于移除大小为 m的子集，否则继续生成m+1大小的频繁项集
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 对于新生成的m+1大小的频繁项集，计算新生成的关联规则的右则的集合
        if len(Hmp1) > 1:  # 如果不止一条规则满足要求（新生成的关联规则的右则的集合的大小大于1），进一步递归合并，
            # 这样做的结果就是会有“[1|多]->多”(右边只会是“多”，因为合并的本质是频繁子项集变大，
            # 而calcConf函数的关联结果的右侧就是频繁子项集）的关联结果
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    '''
        根据频繁项集和最小可信度生成规则。
        L(list):存储频繁项集
        supportData(dict):存储着所有项集（不仅仅是频繁项集）的支持度
        minConf(float):最小可信度
    '''
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:  # 对于每一个频繁项集的集合freqSet
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:  # 如果频繁项集中的元素个数大于2，需要进一步合并，这样做的结果就是会有“[1|多]->多”(右边只会是“多”，
                # 因为合并的本质是频繁子项集变大，而calcConf函数的关联结果的右侧就是频繁子项集），的关联结果
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    sorted(bigRuleList)
    return bigRuleList


myDat = loadDataSet()  # 导入数据集






dataset = loadDataSet()

if __name__ == '__main__':

    '''
    调用find_frequent_itemsets()生成频繁项
    @:param minimum_support表示设置的最小支持度，即若支持度大于等于inimum_support，保存此频繁项，否则删除
    @:param include_support表示返回结果是否包含支持度，若include_support=True，返回结果中包含itemset和support，否则只返回itemset
    '''
    frequent_itemsets = fpg.find_frequent_itemsets(dataset, minimum_support=50, include_support=True)
    print(type(frequent_itemsets))   # print type

    frequent_itemsets =list(frequent_itemsets)


    count =len(frequent_itemsets)
    # for a, b in tmp:
    #     count+=1
    L = [[] for i in range(count)]
    supportData ={}
    for itemset, support in frequent_itemsets:    # 将generator结果存入list

        supportData[frozenset(itemset)] = support/count
        L[len(itemset)-1].append(frozenset(itemset))

    print(L,supportData)
    rules = generateRules(L, supportData, minConf=0.5)
    print('rules:\n', rules)