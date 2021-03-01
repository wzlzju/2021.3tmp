import datetime
import random
from random import choice
import query

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
        self.conditionType = conditionType
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
        if len(self.result) > 10:
            self.groupingFlag = 1
            self.grouping()
        self.preprocess()
        self.possible_children = self.allChlidren()
        self.children_indices = [-1 for _ in range(len(self.possible_children))]
        self.times = 0
        self.profit = 0.0
        self.ppt = 0.0

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

    def selectSubRoot(self):
        pass

    def selectNode(self):
        pass

    def simulation(self, selected, init_depth):
        pass
        # result = 0
        # cdepth = init_depth
        # cnode = self.nodesList[selected]
        # while(True):
        #     if cdepth > self.depthL:
        #         break
        #     # random select a node
        #     cnode = randomSelectChild(cnode)
        #     # calculate current profit
        #     cprofit = 0
        #     result += self.decay**(cdepth-init_depth)*cprofit
        # return result


    def backpropagation(self, current, result):
        pass
        # while(True):
        #     if current == -1:
        #         break
        #     self.nodesList[current].times += 1
        #     self.nodesList[current].profit += result
        #     self.nodesList[current].ppt = self.nodesList[current].profit/self.nodesList[current].times
        #     current = self.nodesParent[current]
        #     result *= self.gamma

    def recordsDecay(self):
        """
        when beginning a new search, sys need to decay times and profit in every existed node
        :return:
        """
        for node in self.nodesList:
            node.times *= self.decay
            node.profit *= self.decay