# frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。
import numpy as np
import pandas as pd



class node:
    def __int__(self):
        self.items = []
        self.key = 0
        self.sup = 0
    def show(self):
        print(self.items, bin(self.key), end=" ")
        try:
            print(self.sup)
        except:
            print()
            pass


def ListToNode(items):
    nodeT = node()
    nodeT.key =0
    nodeT.items = items
    for index,i in enumerate(defaultSL): # 进行编码
        if i in items:
            nodeT.key += 1 << (len(defaultSL) - index -1)
    return nodeT


def keyToList(key):
    ans =[]
    global defaultSL
    for i in range(1,len(defaultSL) +1):
        if key  & (1 << (len(defaultSL)) - i) == (1 << (len(defaultSL)) - i): # 含有第i个
            ans.append(defaultSL[i-1])
    return ans
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
    # print(data.head())

    return np.array(data['Apriori'])


# 返回只有单个元素的候选集
def createC1(dataSet):
    '''
        构建初始候选项集的列表，即所有候选项集只包含一个元素，
        C1是大小为1的所有候选项集的集合
    '''
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    # return map( frozenset, C1 )
    # return [var for var in map(frozenset,C1)]
    return [frozenset(var) for var in C1]


def scanD(D, Ck, minSupport):
    '''
        计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
        返回满足最小支持度的项集的集合，和所有项集支持度信息的字典。
    '''

    for i in Ck:
        i.sup =0
        for item in D:
            if i.key & item.key == i.key: #与运算判断子集
                i.sup +=1
    ans =[]
    for i in Ck:
        i.sup = i.sup / len(D)
        if i.sup >= minSupport:
            ans.append(i)
            # i.show()
    return ans



def aprioriGen(Lk, k):  # Aprior算法
    '''
        由初始候选项集的集合Lk生成新的生成候选项集，
        k表示生成的新项集中所含有的元素个数
        注意其生成的过程中，首选对每个项集按元素排序，然后每次比较两个项集，只有在前k-1项相同时才将这两项合并。这样做是因为函数并非要两两合并各个集合，那样生成的集合并非都是k+1项的。在限制项数为k+1的前提下，只有在前k-1项相同、最后一项不相同的情况下合并才为所需要的新候选项集。
    '''

    global twoDif

    retList = []
    lenLk = len(Lk)
    nowSet = set()
    for i in range(lenLk):
        for j in range(i + 1, lenLk):

            a = Lk[i]
            b = Lk[j]
            t = a.key ^ b.key
            #要注意候选集以及去重的问题
            if t in twoDif:

                nodeT = node()
                nodeT.key = a.key | b.key
                if nodeT.key not in nowSet:
                    nowSet.add(nodeT.key)
                    # a.show()
                    # b.show()
                    nodeT.items = keyToList(nodeT.key)
                    retList.append(nodeT)
                else:
                    pass

    return retList


def apriori(dataSet, minSupport=0.5):
    """
    该函数为Apriori算法的主函数，按照前述伪代码的逻辑执行。Ck表示项数为k的候选项集，最初的C1通过createC1()函数生成。Lk表示项数为k的频繁项集，supK为其支持度，Lk和supK由scanD()函数通过Ck计算而来。
    :param dataSet:
    :param minSupport:
    :return:
    """
    # C1 = createC1(dataSet)  # 构建初始候选项集C1  [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    global defaultSL
    global defaultS

    print(defaultSL)
    C1 = [ListToNode([i]) for i in defaultSL]


    # D = [set(var) for var in dataSet]  # 集合化数据集

    F1 = scanD(dataSet, C1, minSupport)  # 构建初始的频繁项集，即所有项集只有一个元素

    # print()
    L = [F1]
    k = 2  # 项集应该含有2个元素，所以 k=2

    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k) # 计算候选集
        # print("C",len(L[k - 2]),k)
        # for i in Ck:
        #     i.show()
        Fk= scanD(dataSet, Ck, minSupport) # 筛选最小支持度的频繁项集

        L.append(Fk)  # 将符合最小支持度要求的项集加入L
        k += 1  # 新生成的项集中的元素个数应不断增加
    return L  # 返回所有满足条件的频繁项集的列表，和所有候选项集的支持度信息


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



defaultS = set()  # 存储所有单项的集合
myDat = loadDataSet()  # 导入数据集
for i in myDat:
    for j in i:
        if j not in defaultS:
            defaultS.add(j)
defaultSL = list(defaultS)
defaultSL = sorted(defaultSL) # 把所有单项计算出来并排序，形成默认顺序，方便后面进行二进制编码
print(defaultSL)


twoDif =set() # 计算仅仅两个不同时的二进制集合
length = len(defaultSL)
for i in range(length):
    for j in range(length):
        if i != j:
            t=0
#             print(i,j)
            t = 1<<(i)
            t+=1<<(j)
#             print(bin(t))
            twoDif.add(t)
# for i in twoDif:
#     print(bin(i))
Items = []



#对项集进行二进制编码
for items in myDat:
    nodeT = node()
    nodeT.key =0
    nodeT.items = items
    for index,i in enumerate(defaultSL): # 进行编码
        if i in items:
            nodeT.key += 1 << (len(defaultSL) - index -1)
    Items.append(nodeT)
    # print(items, bin(nodeT.key))



L = apriori(Items, 0.05)  # 选择频繁项集

print(u"频繁项集L：")
for li in L:
    for i in li:
        i.show()

#
# rules = generateRules(L, suppData, minConf=0.5)
# print('rules:\n', rules)