import datetime
import random
from random import choice
import query
import numpy as np
import math
import copy
from api import API
from tools import *


class metadata(object):
    def __init__(self):
        self.sourceName = ["mobileTraj", "taxiTraj", "weibo", "poi"]
        self.newSourceName = ["people", "car", "weibo", "poi"]
        self.sourceNum = len(self.newSourceName)
        self.source = list(range(self.sourceNum))
        self.sourceDataType = ["traj", "traj", "point", "point"]
        self.sourceDataNSTAttr = [[], [], [], []]
        self.sourceSTType = ["st", "st", "sta", "sa"]
        self.dataNum = [3799, 1000, 7350, 5000]
        self.conditionTypeNum = 3
        self.decay=0.1
        self.initialization()
    
    def initialization(self):
        self.recordData = [set(), set(), set(), set()]
        self.tmpRecordData = [set(), set(), set(), set()]
        self.recordStcubes = set()
    
    def updateResultData(self, source, resultList):
        # print('updateResultData')
        dataIdSet = self.recordData[source]
        dataIdList = []
        for result in resultList:
            dataIdList.append(result['id'])
            self.recordStcubes = self.recordStcubes.union(result['stcubes'])
        self.recordData[source] = dataIdSet.union(dataIdList)
        return len(self.recordData[source]) - len(dataIdSet)
    
    def updateTmpData(self, source, resultList):
        # print('updateTmpData:', len(self.tmpRecordData), source)
        # print(len(self.tmpRecordData[source]))
        dataSet = self.tmpRecordData[source]
        resultIdList = [result['id'] for result in resultList]
        self.tmpRecordData[source] = dataSet.union(resultIdList)
        return len(self.tmpRecordData[source]) - len(dataSet)
    
    def copyFromRecord(self):
        self.tmpRecordData = copy.deepcopy(self.recordData)


meta = metadata()


class queryNode(object):
    def __init__(self, **kwargs):
        self.source = kwargs.get('source')
        self.numConditionType = 2
        self.queryObj = kwargs.get('queryObj')
        self.queryFrom = kwargs.get('queryFrom')
        self.conditionDict = kwargs.get('conditionDict')

        if self.source in meta.newSourceName:
            self.source = meta.newSourceName.index(self.source)
        if 'time' in self.conditionDict.keys():
            if 'geo' in self.conditionDict.keys():
                self.conditionType = 0
            else:
                self.conditionType = 1
        else:
            self.conditionType = 2
        
        printInfo = ''
        if self.queryObj is None:
            self.result = None
        elif self.queryFrom is 'user': 
            self.dataIdFromFather = None
            self.result = self.queryObj.queryIdxSimplify(self.source, self.conditionDict)
            newDataNum = meta.updateResultData(self.source, self.result)
        else:
            self.dataIdFromFather = kwargs.get('dataIdFromFather')
            self.sourceFromFather = kwargs.get('sourceFromFather')
            self.scubeList = kwargs.get('scubeList')
            self.result = self.queryObj.queryByDataId(
                self.dataIdFromFather, self.sourceFromFather, self.source, self.conditionType)
            newDataNum = meta.updateTmpData(self.source, self.result)
            printInfo += 'sourceFromFather: ' + str(self.sourceFromFather) + ' dataIdFromFather: ' + self.dataIdFromFather
        
        self.groupingFlag = 0
        self.resultG = None
        if len(self.result) > 100000: 
            self.groupingFlag = 1
            self.grouping()
        # self.preprocess()
        self.possible_children = self.allChlidren()
        self.children_indices = [-1 for _ in range(len(self.possible_children))]
        self.resultLen = len(self.resultG) if self.groupingFlag else len(self.result)
        
        print('source:', self.source, printInfo, 
            'queryFrom:', self.queryFrom, 
            'conditionDict:', self.conditionDict, 
            'resultLen:', self.resultLen, '\n')

        self.ppt = newDataNum

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

    def allChlidren(self):  
        """
        Return: 
            [data/data group(index), T/S/A(0/1/2), source(0/1/2/3...)]
        """
        def getEnumeration(x=0, yMax=[1, 1], zMax=1):
            return [[x, [y_0, y_1], z]
                for y_0 in range(yMax[0]+1) 
                for y_1 in range(yMax[1]+1) 
                for z in range(zMax+1) 
                if y_0 !=0 or y_1 != 0]

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
            ret = [[i, 0, j] for i in range(len(self.result)) for j in range(meta.sourceNum)]
            # for i in range(len(self.result)):
            #     if meta.sourceName[self.source] == "mobileTraj":
            #         # ret.append([i, 0, 0])
            #         # ret.append([i, 0, 1])
            #         # ret.append([i, 0, 2])
            #         # ret.append([i, 1, 0])
            #         # ret.append([i, 1, 1])
            #         # ret.append([i, 1, 2])
            #         ret += getEnumeration(i, [1, 1], 2)
            #         ret.append([i, [0, 1], 3])
            #     elif meta.sourceName[self.source] == "taxiTraj":
            #         # ret.append([i, 0, 0])
            #         # ret.append([i, 0, 1])
            #         # ret.append([i, 0, 2])
            #         # ret.append([i, 1, 0])
            #         # ret.append([i, 1, 1])
            #         # ret.append([i, 1, 2])
            #         ret += getEnumeration(i, [1, 1], 2)
            #         ret.append([i, [0, 1], 3])
            #     elif meta.sourceName[self.source] == "weibo":
            #         # ret.append([i, 0, 0])
            #         # ret.append([i, 0, 1])
            #         # ret.append([i, 0, 2])
            #         # ret.append([i, 1, 0])
            #         # ret.append([i, 1, 1])
            #         # ret.append([i, 1, 2])
            #         ret += getEnumeration(i, [1, 1], 2)
            #         ret.append([i, [0, 1], 3])
            #         #ret.append([i,2,2])
            #         #ret.append([i,2,3])
            #     elif meta.sourceName[self.source] == "poi":
            #         # ret.append([i, 1, 0])
            #         # ret.append([i, 1, 1])
            #         # ret.append([i, 1, 2])
            #         # ret.append([i, 1, 3])
            #         ret += getEnumeration(i, [0, 1], 3)
            #         # ret.append([i,2,2])
            #         # ret.append([i,2,3])
        return ret


class mcts(object):
    def __init__(self, queryObj=None, depthL=10, timeL=1000, gamma=0.1):
        self.queryObj = queryObj
        self.depthL = depthL
        # self.timeL = datetime.timedelta(seconds=timeL)  # in seconds
        self.timeL = int(timeL)
        self.gamma = gamma
        self.initialization()
        self.dataConfig = API().query(url='py/querySTConfig', type="get")
        self.scubeNum = self.dataConfig['scubeNum']
        self.dataIdList = API().query(url='py/queryDataSetIds', type="get")
        self.dataIdDict = dict(zip(self.dataIdList, range(len(self.dataIdList))))
    
    def initialization(self):
        self.nodesList = []  # [queryNode]
        self.rootsList = []  # [idx]
        self.nodesChildren = {}  # {idx: [idx]}
        self.nodesParent = {}  # {idx: idx}
        self.currentNodesFlag = []  # [0/1] len==nodeList
        self.heightList = None  # min distance to leaf node
        self.depthList = None  # distance to root node
        self.pptDict = None
        self.availChildDict, self.target2dataId, self.dataId2profit, self.dataId2coin = {}, {}, {}, {}
        meta.initialization()
    
    def getPpt(self, current):
        childIdx = self.nodesChildren[current]
        if self.currentNodesFlag[current] != 1:
            self.pptDict[current] = self.nodesList[current].ppt
        elif len(childIdx) != 0:
            for idx in childIdx:
                self.getPpt(idx)
    
    def selectChild(self, recommendNum=1):
        self.pptDict = {}
        for root in self.rootsList:
            self.getPpt(root)
        
        start = datetime.datetime.utcnow()
        self.nodeId2value = {}
        for nodeId in self.pptDict.values():
            cnode = self.nodesList[nodeId]
            for child in cnode.possible_children:
                end = datetime.datetime.utcnow()
                if end - start >= datetime.timedelta(seconds=60):
                    break
                resultIdx, conditionType, selectSource = child
                conditionDict = self.getConditionDict(
                    conditionType, cnode.result[resultIdx]['bbx'])
                childNodeId = self.constructNewNodefromChild(nodeId, child={'source': selectSource,
                    'dataIdFromFather': cnode.result[resultIdx]['id'],
                    'sourceFromFather': cnode.source,
                    'scubeList': cnode.result[resultIdx]['scube'],
                    'conditionDict': conditionDict}
                )
                self.nodeId2value[childNodeId] = self.nodesList[childNodeId].resultLen
        
        nodeIdSortByValue = sorted(self.nodeId2value.items(), key=lambda item:item[1], reverse=True)
        nodeIdSortList = [item[0] for item in nodeIdSortByValue]
        recommendList = [{'id': idx, 
            'father': self.nodesParent[idx], 
            'source': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].source], 
            'sourceFromFather': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].sourceFromFather], 
            'resultLen': self.nodesList[idx].resultLen,
            'dataid': self.nodesList[idx].dataIdFromFather,
            'mode': self.nodesList[idx].conditionType,
            'sqlobject': self.nodesList[idx].conditionDict}
            for idx in nodeIdSortList[: recommendNum]]
        return recommendList

    def constructNewNodefromChild(self, pid, child):
        """
        sys uses this func to expand new nodes
        :param pid: parent node idx
        :param child: child obj returned from func allChildren()
        :return: new node idx
        """
        cid = len(self.nodesList)
        # print('child:', child)
        self.nodesList.append(queryNode(source=child['source'],
                                        dataIdFromFather=child['dataIdFromFather'],
                                        sourceFromFather=child['sourceFromFather'],
                                        scubeList=child['scubeList'], 
                                        conditionDict=child['conditionDict'],
                                        queryFrom='sys',
                                        queryObj=self.queryObj))
        
        self.nodesParent[cid] = int(pid)
        self.nodesChildren[cid] = []
        self.nodesChildren[pid] += [cid]
        p = self.nodesList[pid]
        # p.children_indices[p.possible_children.index(child)] = cid
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
        # 从数据id转成result中的索引
        qdataIdx, scubeList = None, None
        for idx, item in enumerate(p.result):
            if qdata == item["id"]: 
                qdataIdx = idx
                scubeList = item['scube']
        if qdataIdx is None or scubeList is None:
            qdataIdx = len(p.result)
            scubeList = [1]
        
        cid = self.constructNewNodefromChild(qnode, child={
            'source': qsource,
            'dataIdFromFather': qdata,
            'sourceFromFather': p.source,
            'scubeList': scubeList,
            'conditionDict': qattr})
        
        typeFlag = {0: [1, 1], 1: [1, 0], 2: [0, 1]}[self.nodesList[cid].conditionType]
        childList = [qdataIdx, typeFlag, qsource]
        p.possible_children.append(childList)
        p.children_indices.append(cid)
        self.confirmNode(cid)
        return cid

    def constructNewNodefromCondition(self, conditionDict, qsource):
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
            conditionDict=conditionDict,
            queryObj=self.queryObj,
            queryFrom='user'
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
        cnode = self.nodesList[cidx]
        meta.updateResultData(cnode.source, cnode.result)
    
    def getConditionDict(self, conditionType, fullCondition):
        conditionDict = {}
        conditionTypeList = {0: [1, 1], 1: [1, 0], 2: [0, 1]}[conditionType]
        for idx, val in enumerate(conditionTypeList):
            if val == 1:
                conditionType = ['time', 'geo', 'TBD'][idx]
                conditionDict[conditionType] = fullCondition[conditionType]
        return conditionDict
    
    def recommendLog(self):
        logList = [
            {
                'source': node.source, 
                'condition': node.conditionDict,
                'resultLen:': node.resultLen,
                'profQ': node.profQ, 
                'times': node.times,
                'profit': node.profit,
                'ppt': node.ppt
            } 
            for node in self.nodesList
        ]
        return logList



if __name__ == "__main__":
    m = mcts(query.queryObj())
    mBeta = mcts(query.queryObj())
    # conditionDict = {
    #     'geo': [120.69783926010132, 120.69760859012602, 28.013147821134936, 28.012896816979197],
    #     'time': ["07:00:00", "08:00:00"]
    # }
    # m.constructNewNodefromCondition(conditionDict, 0)
    # for t in range(100):
    #     recommendList = m.nodesRecommend(recommendNum=10)
    #     choiceId = recommendList[0]['id']
    #     print('choiceId:', choiceId)
    #     m.confirmNode(choiceId)
    #     # 要记录：
    #     # 推荐前10的节点id，nodeList
    #     recommendIdList = [item['id'] for item in recommendList]
    #     saveJson({'recommendId': recommendIdList, 'currentNodesFlag': m.currentNodesFlag, 
    #         'nodesParent': m.nodesParent, 'nodeList': m.recommendLog()}, 'log')
    for t in range(256):
        queryCondition = API().query(type='get', url='py/mockReq')
        print(queryCondition)
        m.initialization()
        m.constructNewNodefromCondition(queryCondition['attr'], queryCondition['source'])
        for _ in range(10):
            mBeta = copy.deepcopy(m)
            trainItem, recommendList = mBeta.DRLTrain()
            saveJson(trainItem, 'train')

            choiceId = recommendList[0]['id']
            choiceNode = mBeta.nodesList[choiceId]
            fatherId = mBeta.nodesParent[choiceId]

            # Updata m using mBeta
            cid = m.constructNewNodefromQuery(
                fatherId, 
                choiceNode.dataIdFromFather, 
                choiceNode.conditionDict, 
                choiceNode.source)
            m.confirmNode(cid)
            
            for i in range(len(m.nodesList)):
                if i == cid:
                    m.nodesList[i].profit = mBeta.nodesList[choiceId].profit * meta.decay
                    m.nodesList[i].times = mBeta.nodesList[choiceId].times * meta.decay
                    m.nodesList[i].ppt = mBeta.nodesList[choiceId].ppt
                else:
                    m.nodesList[i].profit = mBeta.nodesList[i].profit * meta.decay
                    m.nodesList[i].times = mBeta.nodesList[i].times * meta.decay
                    m.nodesList[i].ppt = mBeta.nodesList[i].ppt
            
            print('========================================')
            print('choice node:', 
                'source:', choiceNode.source, 
                'conditionDict:', choiceNode.conditionDict, 
                'resultLen:', choiceNode.resultLen)
            print('========================================\n')