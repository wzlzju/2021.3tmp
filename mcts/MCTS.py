import datetime
import random
from random import choice
import query
import numpy as np

class metadata(object):
    def __init__(self):
        self.sourceNum = 3
        self.source = list(range(self.sourceNum))
        self.sourceName = ["mobileTraj","taxiTraj","weibo"]
        self.sourceDataType = ["traj","traj","point"]
        self.sourceDataNSTAttr = [[],[],[]]
        self.sourceSTType = ["st","st","st"]
        self.dataNum = [10000,10000,10000]

meta = metadata()

class queryNode(object):
    def __init__(self, source=None, conditionType=None, condition=None, queryObj=None):
        self.source = source
        self.conditionType = conditionType if type(conditionType) is str else {0:"T",1:"S",2:"A"}[conditionType]
        self.condition = condition
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
        if len(self.result) > 1000:
            self.groupingFlag = 1
            self.grouping()
        self.preprocess()
        self.possible_children = self.allChlidren()
        self.children_indices = [-1 for _ in range(len(self.possible_children))]
        self.profQ = len(self.resultG) if self.groupingFlag else len(self.result)
        self.times = 0
        self.profit = 0.0
        self.profitHistory = []
        self.ppt = 1.0

    def grouping(self):
        self.resultG = []
        if meta.sourceDataType[self.source] == "point":
            bbox = self.queryObj.bboxp2(meta.sourceName[self.source],self.result)
        elif meta.sourceDataType[self.source] == "traj":
            bbox = self.queryObj.bboxt2(meta.sourceName[self.source],self.result)
        l,r,u,d = bbox
        clng = (l+r)/2
        clat = (u+d)/2
        g1 = []         #   1   |   2
        g2 = []         # -----+-----
        g3 = []         #  3  |    4
        g4 = []
        g=[g1,g2,g3,g4]
        for r in self.result:
            if meta.sourceDataType[self.source] == "point":
                if self.queryObj.pinbbox(meta.sourceName[self.source],r,[l,clng,u,clat]):
                    g1.append(r)
                elif self.queryObj.pinbbox(meta.sourceName[self.source],r,[clng,r,u,clat]):
                    g2.append(r)
                elif self.queryObj.pinbbox(meta.sourceName[self.source],r,[l,clng,clat,d]):
                    g3.append(r)
                elif self.queryObj.pinbbox(meta.sourceName[self.source],r,[clng,r,clat,d]):
                    g4.append(r)
            if meta.sourceDataType[self.source] == "traj":
                if self.queryObj.tinbbox(meta.sourceName[self.source],r,[l,clng,u,clat]):
                    g1.append(r)
                elif self.queryObj.tinbbox(meta.sourceName[self.source],r,[clng,r,u,clat]):
                    g2.append(r)
                elif self.queryObj.tinbbox(meta.sourceName[self.source],r,[l,clng,clat,d]):
                    g3.append(r)
                elif self.queryObj.tinbbox(meta.sourceName[self.source],r,[clng,r,clat,d]):
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
                    self.SQC.append(self.queryObj.bboxp2(meta.sourceName[self.source],rg))
            else:
                for r in self.result:
                    self.SQC.append(self.queryObj.bboxp(meta.sourceName[self.source],r))
        elif meta.sourceName[self.source] == "mobileTraj":
            self.SQC = []
            self.TQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(self.queryObj.bboxt2(meta.sourceName[self.source],rg))
                    self.TQC.append(self.queryObj.tboxt2(meta.sourceName[self.source],rg))
            else:
                for r in self.result:
                    self.SQC.append(self.queryObj.bboxt(meta.sourceName[self.source],r))
                    self.TQC.append(self.queryObj.tboxt(meta.sourceName[self.source],r))
        elif meta.sourceName[self.source] == "taxiTraj":
            self.SQC = []
            self.TQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(self.queryObj.bboxt2(meta.sourceName[self.source],rg))
                    self.TQC.append(self.queryObj.tboxt2(meta.sourceName[self.source],rg))
            else:
                for r in self.result:
                    self.SQC.append(self.queryObj.bboxt(meta.sourceName[self.source],r))
                    self.TQC.append(self.queryObj.tboxt(meta.sourceName[self.source],r))
        elif meta.sourceName[self.source] == "weibo":
            self.SQC = []
            self.TQC = []
            if self.groupingFlag == 1:
                for rg in self.resultG:
                    self.SQC.append(self.queryObj.bboxp2(meta.sourceName[self.source],rg))
                    self.TQC.append(self.queryObj.tboxp2(meta.sourceName[self.source],rg))
            else:
                for r in self.result:
                    self.SQC.append(self.queryObj.bboxp(meta.sourceName[self.source],r))
                    self.TQC.append(self.queryObj.tboxp(meta.sourceName[self.source],r))
        self.QCs = [self.TQC, self.SQC, self.AQC]

    def allChlidren(self):  # return: [data/data group(index), T/S/A(0/1/2), source(0/1/2/3...)]
        ret = []
        if self.groupingFlag == 1:
            for i in range(len(self.resultG)):
                if meta.sourceName[self.source] == "mobileTraj":
                    ret.append([i,0,0])
                    ret.append([i,0,1])
                    ret.append([i,0,2])
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    #ret.append([i,1,3])
                elif meta.sourceName[self.source] == "taxiTraj":
                    ret.append([i,0,0])
                    ret.append([i,0,1])
                    ret.append([i,0,2])
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    #ret.append([i,1,3])
                elif meta.sourceName[self.source] == "weibo":
                    ret.append([i,0,0])
                    ret.append([i,0,1])
                    ret.append([i,0,2])
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    #ret.append([i,1,3])
                elif meta.sourceName[self.source] == "poi":
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    ret.append([i,1,3])
                    #ret.append([i,2,2])
                    #ret.append([i,2,3])
        else:
            for i in range(len(self.result)):
                if meta.sourceName[self.source] == "mobileTraj":
                    ret.append([i,0,0])
                    ret.append([i,0,1])
                    ret.append([i,0,2])
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    #ret.append([i,1,3])
                elif meta.sourceName[self.source] == "taxiTraj":
                    ret.append([i,0,0])
                    ret.append([i,0,1])
                    ret.append([i,0,2])
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    #ret.append([i,1,3])
                elif meta.sourceName[self.source] == "weibo":
                    ret.append([i,0,0])
                    ret.append([i,0,1])
                    ret.append([i,0,2])
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    #ret.append([i,1,3])
                    #ret.append([i,2,2])
                    #ret.append([i,2,3])
                elif meta.sourceName[self.source] == "poi":
                    ret.append([i,1,0])
                    ret.append([i,1,1])
                    ret.append([i,1,2])
                    ret.append([i,1,3])
                    #ret.append([i,2,2])
                    #ret.append([i,2,3])
        return ret


class mcts(object):
    def __init__(self, queryObj=None, depthL=10, timeL=10.0, gamma=0.1, decay=0.1):
        self.queryObj = queryObj
        self.nodesList = []     # [queryNode]
        self.rootsList = []     # [idx]
        self.nodesChildren = {}     # {idx: [idx]}
        self.nodesParent = {}       # {idx: idx}
        self.currentNodesFlag = []      # [0/1] len==nodeList
        self.depthL = depthL
        self.timeL = datetime.timedelta(seconds=timeL)      # in seconds
        self.gamma = gamma
        self.decay = decay
        self.heightList = None  # min distance to leaf node
        self.depthList = None  # distance to root node

    def nodesRecommand(self):
        startT = datetime.datetime.utcnow()
        self.recordsDecay()
        while(True):
            croot, cdepth = self.selectSubRoot()
            cselected = self.selectNode(croot)
            cresult = self.simulation(cselected, cdepth)
            self.backpropagation(cselected, cresult)
            endT = datetime.datetime.utcnow()
            if endT-startT >= self.timeL:
                break
        # 排名

    def constructNewNodefromChild(self, pid, child):
        """
        sys uses this func to expand new nodes
        :param pid: parent node idx
        :param child: child obj returned from func allChildren()
        :return: new node idx
        """
        cid = len(self.nodesList)
        source = child[2]
        conditionType = {0:"T",1:"S",2:"A"}[child[1]]
        condition = self.nodesList[pid].QCs[child[1]][child[0]]
        self.nodesList.append(queryNode(source=source,
                                        conditionType=conditionType,
                                        condition=condition,
                                        queryObj=self.queryObj))
        self.nodesParent[cid] = pid
        p = self.nodesList[pid]
        p.children_indices[p.possible_children.index(child)] = cid
        self.nodesChildren[cid] = []
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
        child = [qdata, qattr, qsource]
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
        conditionType = 1
        condition = [120.3551055, 120.9374903, 28.13387079, 27.876454730000003]
        qsource = 0
        cid = len(self.nodesList)
        source = qsource
        rootQueryNode = queryNode(source=source,
                                    conditionType=conditionType,
                                    condition=condition,
                                    queryObj=self.queryObj)
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
                self.heightList[current] = min(self.getNodeHeight(idx)+1, self.heightList[current])
        return self.heightList[current]
    
    def getNodeDepth(self, current, cdepth):
        self.depthList[current] = cdepth
        childIdx = self.nodesChildren[current]
        if len(childIdx) != 0:
            for idx in childIdx:
                self.getNodeDepth(idx, cdepth+1)

    def selectSubRoot(self):
        nodeNum = len(self.nodesList)
        self.heightList = np.array([10] * nodeNum)
        self.depthList = np.array([0] * nodeNum)
        for root in self.rootsList:
            self.getNodeHeight(root)
            self.getNodeDepth(root, 0)
        # height normalization
        normHeightList = self.heightList / np.sum(self.heightList)
        pptList = [node.ppt for node in self.nodesList]
        importList = [normHeightList[i] * pptList[i] for i in range(len(pptList))]
        # sample node based on importance of nodes
        sampleIdx = np.random.choice(list(range(nodeNum)), 1, p=importList)[0]
        return self.nodesList[sampleIdx], self.depthList[sampleIdx]

    def selectNode(self, croot):
        current = croot
        while -1 not in self.nodesList[current].children_indices:
            maxProfit = 0
            for childIdx in self.nodesList[current].children_indices:
                if self.nodesList[childIdx].profit > maxProfit:
                    maxProfit = self.nodesList[childIdx].profit
                    current = childIdx
        return current
    
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
        while(True):
            if relativeDepth > self.depthL:
                break
            # random select a node
            cnode = self.randomSelectChild(cnode)
            # calculate current profit
            result += self.gamma**(relativeDepth) * cnode.profit
            relativeDepth += 1
        return result

    def backpropagation(self, current, result):
        while(True):
            if current == -1:
                break
            cprofit = self.nodesList[current].profQ
            childIdx = self.nodesChildren[current]
            for idx in childIdx:
                cprofit += self.gamma * self.nodesList[idx].profit / len(childIdx)
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

if __name__ == "__main__":
    m = mcts(query.queryObj())
    m.constructNewNodefromCondition(1, [120.3551055, 120.6374903, 28.00387079, 27.876454730000003],0)
    # print(m.nodesList)
    # print(m.rootsList)
    # print(m.nodesList[0].groupingFlag)
    # print(len(m.nodesList[0].result),m.nodesList[0].result)
    # if m.nodesList[0].groupingFlag:
    #     print(len(m.nodesList[0].resultG),m.nodesList[0].resultG)
    # print(len(m.nodesList[0].possible_children),m.nodesList[0].possible_children)
    # print(len(m.nodesList[0].children_indices),m.nodesList[0].children_indices)

    print(m.simulation(0, 10))

