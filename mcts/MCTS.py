import datetime
import random
from random import choice
import query
import numpy as np
import math
from tools import *


class metadata(object):
    def __init__(self):
        self.sourceNum = 3
        self.source = list(range(self.sourceNum))
        self.sourceName = ["mobileTraj", "taxiTraj", "weibo", "poi"]
        self.sourceDataType = ["traj", "traj", "point", "point"]
        self.sourceDataNSTAttr = [[], [], [], []]
        self.sourceSTType = ["st", "st", "sta", "sa"]
        self.dataNum = [10000, 10000, 10000, 10000]


meta = metadata()


class queryNode(object):
    def __init__(self, source=None, conditionType=None, condition=None, queryObj=None):
        self.source = source #
        self.conditionType = (
            conditionType
            if type(conditionType) is str
            else {0: "T", 1: "S", 2: "A"}[conditionType]
        )
        self.condition = condition #
        self.queryObj = queryObj
        if queryObj:
            if self.conditionType == "S":
                self.result = queryObj.queryIdxSimplify(source, sRange=condition)
            elif self.conditionType == "T":
                self.result = queryObj.queryIdxSimplify(source, tRange=condition)
            elif self.conditionType == "A":
                pass
        else:
            self.result = None
        self.groupingFlag = 0
        self.resultG = None
        if len(self.result) > 100000:
            self.groupingFlag = 1
            self.grouping()
        self.preprocess()
        self.possible_children = self.allChlidren()
        self.children_indices = [-1 for _ in range(len(self.possible_children))] #
        self.resultLen = len(self.resultG) if self.groupingFlag else len(self.result) #
        print('source:', self.source,  'conditionType:', self.conditionType,
            'condition:', self.condition, 'resultLen:', self.resultLen)
        self.profQ = abs(math.log(self.resultLen/10000, 2)) #
        self.times = 0 #
        self.profit = 0.0 #
        self.profitHistory = [] #
        self.ppt = 1.0 #

    def grouping(self):
        self.resultG = []
        if meta.sourceDataType[self.source] == "point":
            bbox = self.queryObj.bboxp2(meta.sourceName[self.source], self.result)
        elif meta.sourceDataType[self.source] == "traj":
            bbox = self.queryObj.bboxt2(meta.sourceName[self.source], self.result)
        l, r, u, d = bbox
        clng = (l + r) / 2
        clat = (u + d) / 2
        g1 = []  #   1   |   2
        g2 = []  # -----+-----
        g3 = []  #  3  |    4
        g4 = []
        g = [g1, g2, g3, g4]
        for r in self.result:
            if meta.sourceDataType[self.source] == "point":
                if self.queryObj.pinbbox(
                    meta.sourceName[self.source], r, [l, clng, u, clat]
                ):
                    g1.append(r)
                elif self.queryObj.pinbbox(
                    meta.sourceName[self.source], r, [clng, r, u, clat]
                ):
                    g2.append(r)
                elif self.queryObj.pinbbox(
                    meta.sourceName[self.source], r, [l, clng, clat, d]
                ):
                    g3.append(r)
                elif self.queryObj.pinbbox(
                    meta.sourceName[self.source], r, [clng, r, clat, d]
                ):
                    g4.append(r)
            if meta.sourceDataType[self.source] == "traj":
                if self.queryObj.tinbbox(
                    meta.sourceName[self.source], r, [l, clng, u, clat]
                ):
                    g1.append(r)
                elif self.queryObj.tinbbox(
                    meta.sourceName[self.source], r, [clng, r, u, clat]
                ):
                    g2.append(r)
                elif self.queryObj.tinbbox(
                    meta.sourceName[self.source], r, [l, clng, clat, d]
                ):
                    g3.append(r)
                elif self.queryObj.tinbbox(
                    meta.sourceName[self.source], r, [clng, r, clat, d]
                ):
                    g4.append(r)
        if len(g1) > 0:
            self.resultG.append(g1)
        if len(g2) > 0:
            self.resultG.append(g2)
        if len(g3) > 0:
            self.resultG.append(g3)
        if len(g4) > 0:
            self.resultG.append(g4)
        return

    def preprocess(self):
        self.SQC = None
        self.TQC = None
        self.AQC = None
        if meta.sourceName[self.source] == "poi":
            self.SQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(
                        self.queryObj.bboxp2(meta.sourceName[self.source], rg)
                    )
            else:
                for r in self.result:
                    self.SQC.append(r["bbx"]["areaRange"])
        elif meta.sourceName[self.source] == "mobileTraj":
            self.SQC = []
            self.TQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(
                        self.queryObj.bboxt2(meta.sourceName[self.source], rg)
                    )
                    self.TQC.append(
                        self.queryObj.tboxt2(meta.sourceName[self.source], rg)
                    )
            else:
                for r in self.result:
                    self.SQC.append(r["bbx"]["areaRange"])
                    self.TQC.append(r["bbx"]["timeRange"])
        elif meta.sourceName[self.source] == "taxiTraj":
            self.SQC = []
            self.TQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(
                        self.queryObj.bboxt2(meta.sourceName[self.source], rg)
                    )
                    self.TQC.append(
                        self.queryObj.tboxt2(meta.sourceName[self.source], rg)
                    )
            else:
                for r in self.result:
                    self.SQC.append(r["bbx"]["areaRange"])
                    self.TQC.append(r["bbx"]["timeRange"])
        elif meta.sourceName[self.source] == "weibo":
            self.SQC = []
            self.TQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(
                        self.queryObj.bboxp2(meta.sourceName[self.source], rg)
                    )
                    self.TQC.append(
                        self.queryObj.tboxp2(meta.sourceName[self.source], rg)
                    )
            else:
                for r in self.result:
                    self.SQC.append(r["bbx"]["areaRange"])
                    self.TQC.append(r["bbx"]["timeRange"])
        self.QCs = [self.TQC, self.SQC, self.AQC]

    def allChlidren(
        self,
    ):  # return: [data/data group(index), T/S/A(0/1/2), source(0/1/2/3...)]
        ret = []
        if self.groupingFlag == 1:
            for i in range(len(self.resultG)):
                if meta.sourceName[self.source] == "mobileTraj":
                    ret.append([i, 0, 0])
                    ret.append([i, 0, 1])
                    ret.append([i, 0, 2])
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    # ret.append([i,1,3])
                elif meta.sourceName[self.source] == "taxiTraj":
                    ret.append([i, 0, 0])
                    ret.append([i, 0, 1])
                    ret.append([i, 0, 2])
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    # ret.append([i,1,3])
                elif meta.sourceName[self.source] == "weibo":
                    ret.append([i, 0, 0])
                    ret.append([i, 0, 1])
                    ret.append([i, 0, 2])
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    # ret.append([i,1,3])
                elif meta.sourceName[self.source] == "poi":
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    ret.append([i, 1, 3])
                    # ret.append([i,2,2])
                    # ret.append([i,2,3])
        else:
            for i in range(len(self.result)):
                if meta.sourceName[self.source] == "mobileTraj":
                    ret.append([i, 0, 0])
                    ret.append([i, 0, 1])
                    ret.append([i, 0, 2])
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    ret.append([i,1,3])
                elif meta.sourceName[self.source] == "taxiTraj":
                    ret.append([i, 0, 0])
                    ret.append([i, 0, 1])
                    ret.append([i, 0, 2])
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    ret.append([i,1,3])
                elif meta.sourceName[self.source] == "weibo":
                    ret.append([i, 0, 0])
                    ret.append([i, 0, 1])
                    ret.append([i, 0, 2])
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    ret.append([i,1,3])
                    #ret.append([i,2,2])
                    #ret.append([i,2,3])
                elif meta.sourceName[self.source] == "poi":
                    ret.append([i, 1, 0])
                    ret.append([i, 1, 1])
                    ret.append([i, 1, 2])
                    ret.append([i, 1, 3])
                    # ret.append([i,2,2])
                    # ret.append([i,2,3])
        return ret


class mcts(object):
    def __init__(self, queryObj=None, depthL=10, timeL=10.0, gamma=0.1, decay=0.1):
        self.queryObj = queryObj
        self.depthL = depthL
        self.timeL = datetime.timedelta(seconds=timeL)  # in seconds
        self.gamma = gamma
        self.decay = decay
        self.initialization()
    
    def initialization(self):
        self.nodesList = []  # [queryNode]
        self.rootsList = []  # [idx]
        self.nodesChildren = {}  # {idx: [idx]}
        self.nodesParent = {}  # {idx: idx}
        self.currentNodesFlag = []  # [0/1] len==nodeList
        self.heightList = None  # min distance to leaf node
        self.depthList = None  # distance to root node
        self.pptDict = None
    
    def getPpt(self, current):
        childIdx = self.nodesChildren[current]
        if self.currentNodesFlag[current] != 1:
            self.pptDict[current] = self.nodesList[current].ppt
        elif len(childIdx) != 0:
            for idx in childIdx:
                self.getPpt(idx)


    def nodesRecommend(self, recommendNum=3):
        startT = datetime.datetime.utcnow()
        self.recordsDecay()
        while True:
            croot, cdepth = self.selectSubRoot()
            cselected = self.selectNode(croot)
            cresult = self.simulation(cselected, cdepth)
            self.backpropagation(cselected, cresult)
            endT = datetime.datetime.utcnow()
            if endT - startT >= self.timeL:
                break

        self.pptDict = {}
        for root in self.rootsList:
            self.getPpt(root)
        pptSortedList = sorted(self.pptDict.items(), key=lambda item:item[1], reverse=False)
        idxSortedList = [item[0] for item in pptSortedList]
        recommendList = [{'id': idx, 
            'father': self.nodesParent[idx], 
            'source': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].source], 
            'type': self.nodesList[idx].conditionType, 
            'data': self.nodesList[idx].condition}
            for idx in idxSortedList[: recommendNum]]
        return recommendList


    def constructNewNodefromChild(self, pid, child):
        """
        sys uses this func to expand new nodes
        :param pid: parent node idx
        :param child: child obj returned from func allChildren()
        :return: new node idx
        """
        cid = len(self.nodesList)
        source = child[2]
        conditionType = {0: "T", 1: "S", 2: "A"}[child[1]]
        condition = self.nodesList[pid].QCs[child[1]][child[0]]
        self.nodesList.append(
            queryNode(
                source=source,
                conditionType=conditionType,
                condition=condition,
                queryObj=self.queryObj,
            )
        )
        self.nodesParent[cid] = int(pid)
        self.nodesChildren[cid] = []
        self.nodesChildren[pid] += [cid]
        p = self.nodesList[pid]
        p.children_indices[p.possible_children.index(child)] = cid
        self.currentNodesFlag.append(0)
        return cid

    def constructNewNodefromQuery(self, qnode, qdata, qattr, qsource):
        """
        user uses this func to construct a new query from existed query nodes
        ***Assume all nodes user can construct are in the tree
        ***Actually, it is a node confirming process
        :param qnode: from which node
        :param qdata: from which data of qnode
        :param qattr: from which attribute of qattr
        :param qsource: to which source
        :return: new node idx
        """
        p = self.nodesList[qnode]
        print('len(p.result):', len(p.result))
        # 处理一下qdata，从数据id转成result中的索引
        for idx, item in enumerate(p.result):
            if qdata == item["id"]:
                qdataIdx = idx
        qattr = {"T": 0, "S": 1, "A": 2}[qattr]
        child = [qdataIdx, qattr, qsource]
        # qattr处理成整型
        cid = p.children_indices[p.possible_children.index(child)]
        self.confirmNode(cid)
        # cid = len(self.nodesList)
        # source = qsource
        # if type(qattr) is int:
        #     conditionType = {0:"T",1:"S",2:"A"}[qattr]
        # else:
        #     print("Error. check query attr type in constructNewNodefromQuery(). ")
        # condition = self.nodesList[qnode].QCs[qattr][qdata]
        # self.nodesList.append(queryNode(source=source,
        #                                 conditionType=conditionType,
        #                                 condition=condition,
        #                                 queryObj=self.queryObj))
        # self.nodesParent[cid] = qnode
        # self.nodesChildren[cid] = []
        # self.currentNodesFlag.append(0)
        return cid

    def constructNewNodefromCondition(self, conditionType, condition, qsource):
        """
        user uses this func to construct a new root query
        :param conditionType: 0: T, 1: S, 2: A
        :param condition: the condition param in queryNode
        :param qsource: query which source
        :return: new node idx
        """
        cid = len(self.nodesList)
        source = qsource
        rootQueryNode = queryNode(
            source=source,
            conditionType=conditionType,
            condition=condition,
            queryObj=self.queryObj,
        )
        self.nodesList.append(rootQueryNode)
        self.rootsList.append(cid)
        self.nodesParent[cid] = -1
        self.nodesChildren[cid] = []
        self.currentNodesFlag.append(0)
        self.confirmNode(cid)
        return cid

    def confirmNode(self, cidx):
        """
        sys uses this func to confirm a node when user selecting a recommendation
        :param cidx:
        :return:
        """
        self.currentNodesFlag[cidx] = 1

    def getNodeHeight(self, current):
        childIdx = self.nodesChildren[current]
        if len(childIdx) == 0:
            self.heightList[current] = 1
        else:
            for idx in childIdx:
                self.heightList[current] = min(
                    self.getNodeHeight(idx) + 1, self.heightList[current]
                )
        return self.heightList[current]

    def getNodeDepth(self, current, cdepth):
        self.depthList[current] = cdepth
        childIdx = self.nodesChildren[current]
        if len(childIdx) != 0:
            for idx in childIdx:
                self.getNodeDepth(idx, cdepth + 1)

    def selectSubRoot(self):
        nodeNum = len(self.nodesList)
        self.heightList = np.array([10] * nodeNum)
        self.depthList = np.array([0] * nodeNum)
        for root in self.rootsList:
            self.getNodeHeight(root)
            self.getNodeDepth(root, 0)
        # height normalization
        normHeightList = self.heightList / (max(np.max(self.depthList), self.depthL) + 1)
        pptList = [node.ppt for node in self.nodesList]
        importList = [(1 - normHeightList[i]) * pptList[i] * self.currentNodesFlag[i] for i in range(len(pptList))]
        importList = np.array(importList) / (np.sum(importList))
        # print('importList:', importList)
        # sample node based on importance of nodes
        sampleIdx = np.random.choice(list(range(nodeNum)), 1, p=importList)[0]
        return sampleIdx, self.depthList[sampleIdx]

    def selectNode(self, croot):
        current = croot
        childrenIndices = self.nodesList[current].children_indices
        while -1 not in childrenIndices:
            maxProfit = 0
            for childIdx in childrenIndices:
                if self.nodesList[childIdx].profit > maxProfit:
                    maxProfit = self.nodesList[childIdx].profit
                    current = childIdx
            childrenIndices = self.nodesList[current].children_indices
        selectList = [self.nodesList[current].possible_children[i] 
            for i in range(len(childrenIndices)) if childrenIndices[i] == -1]
        selectChild = choice(selectList)
        print('selectChild:', selectChild)
        expandIndice = self.constructNewNodefromChild(current, selectChild)
        return expandIndice

    def randomSelectChild(self, node):
        selectList = [idx for idx in range(len(node.children_indices))
            if node.children_indices[idx] == -1]
        selectIdx = choice(selectList)
        dataIdx, conditionType, source = node.possible_children[selectIdx]
        condition = node.QCs[conditionType][dataIdx]
        # print(source, conditionType, condition)
        selectNode = queryNode(source=source,
                            conditionType=conditionType,
                            condition=condition,
                            queryObj=self.queryObj)
        return selectNode

    def simulation(self, selected, init_depth):
        result = 0
        relativeDepth = 1
        cnode = self.nodesList[selected]
        while True:
            if relativeDepth > self.depthL:
                break
            # random select a node
            cnode = self.randomSelectChild(cnode)
            # calculate current profit
            result += self.gamma ** (relativeDepth) * cnode.profQ
            relativeDepth += 1
        return result

    def backpropagation(self, current, result):
        while True:
            if current == -1:
                break
            cprofit = self.nodesList[current].profQ
            childIdx = self.nodesChildren[current]
            if len(childIdx) == 0:
                cprofit += result
            else:
                for idx in childIdx:
                    cprofit += self.gamma * self.nodesList[idx].profit / len(childIdx)
            self.nodesList[current].profit = cprofit
            self.nodesList[current].profitHistory.append(cprofit)
            self.nodesList[current].ppt = np.mean(self.nodesList[current].profitHistory)
            current = self.nodesParent[current]

    def recordsDecay(self):
        """
        when beginning a new search, sys need to decay times and profit in every existed node
        :return:
        """
        for node in self.nodesList:
            node.times *= self.decay
            node.profit *= self.decay
    
    def recommendLog(self):
        logList = [
            {
                'source': node.source, 
                'condition': node.condition,
                'resultLen:': node.resultLen,
                'profQ': node.profQ, 
                'times': node.times,
                'profit': node.profit,
                'profitHistory': node.profitHistory,
                'ppt': node.ppt
            } 
            for node in self.nodesList
        ]
        return logList



if __name__ == "__main__":
    m = mcts(query.queryObj())
    m.constructNewNodefromCondition(
        1, [120.66627502441406, 120.66387176513672, 28.008345489218808, 28.00561744200495], 0
    )
    # print(m.nodesList)
    # print(m.rootsList)
    # print(m.nodesList[0].groupingFlag)
    # print(len(m.nodesList[0].result),m.nodesList[0].result)
    # if m.nodesList[0].groupingFlag:
    #     print(len(m.nodesList[0].resultG),m.nodesList[0].resultG)
    # print(len(m.nodesList[0].possible_children),m.nodesList[0].possible_children)
    # print(len(m.nodesList[0].children_indices),m.nodesList[0].children_indices)

    # print(m.simulation(0, 10))
    # print(m.nodesRecommend())
    # print(len(m.nodesList))
    # print(m.currentNodesFlag)

    for t in range(100):
        recommendList = m.nodesRecommend(recommendNum=10)
        choiceId = recommendList[0]['id']
        print('choiceId:', choiceId)
        m.confirmNode(choiceId)
        # 要记录：
        # 推荐前10的节点id，nodeList
        recommendIdList = [item['id'] for item in recommendList]
        saveJson({'recommendId': recommendIdList, 'currentNodesFlag': m.currentNodesFlag, 
            'nodesParent': m.nodesParent, 'nodeList': m.recommendLog()}, 'log')