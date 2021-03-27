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

        self.dataConfig = API().query(url='py/querySTConfig', type="get")
        self.scubeNum = self.dataConfig['scubeNum']
        self.dataNum = self.dataConfig['dataSetInfo']['num']
        self.dataIds = self.dataConfig['dataSetInfo']['ids']
        self.dataIdList = []
        for ids in self.dataIds:
            self.dataIdList += ids
        self.dataIdDict = dict(zip(self.dataIdList, range(len(self.dataIdList))))

        self.heatMap = API().query(url='scubeHeatMap', type="get")

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
        recordDataIdList = []
        for result in resultList:
            recordDataIdList.append(result['id'])
            self.recordStcubes = self.recordStcubes.union(result['stcubes'])
        self.recordData[source] = dataIdSet.union(recordDataIdList)
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
        
        oldDataNum = self.resultLen - newDataNum
        if newDataNum - oldDataNum/10 <= 0:
            self.profQ = 0
        elif meta.dataNum[self.source]-len(meta.recordData[self.source]) <= 0:
            self.profQ = 0
        else:
            self.profQ = abs(math.log((newDataNum-oldDataNum/10)/(
                meta.dataNum[self.source]-len(meta.recordData[self.source])), 2))
        
        self.profit, self.times, self.ppt = 0.0, 0, 1.0

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
        self.scubeNum = meta.scubeNum
        self.dataIdList = meta.dataIdList
        self.dataIdDict = meta.dataIdDict
    
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

    def mtcsStep(self):
        i = 0
        while True:
            print('------------ Round', str(i+1), '------------')
            try:
                meta.copyFromRecord()
                croot, cdepth = self.selectSubRoot()
                cselected = self.selectNode(croot)
                if cselected is None: break
                cresult = self.simulation(cselected, cdepth)
                self.backpropagation(cselected, cresult)
            except Exception as e:
                i -= 1
                print('Error:', e)
            i += 1
            if i >= self.timeL: break
    
    def getTargetIdx(self, scube, source, conditionType):
        """
        Get idx in DRL output according to scube, source and conditionType.
        """
        return conditionType + meta.conditionTypeNum * (source + meta.sourceNum * scube)

    def drawHeatMap(self, FatherNodeId=None, caseNodeId=None):
        self.pptDict = {}
        for root in self.rootsList:
            self.getPpt(root)

        geoProfit = [0] * self.scubeNum
        if FatherNodeId is not None:
            for item in self.nodesList[FatherNodeId].result:
                for scubeIdx in item['scube']:
                    geoProfit[scubeIdx] += 10
            print('sum of heat after children results:', sum(geoProfit))
        
        print('self.pptDict:', self.pptDict)
        print('caseNodeId', caseNodeId)
        if caseNodeId is not None and caseNodeId in self.pptDict.keys():
            print('Case Node id in ppt dict.')
            self.pptDict[caseNodeId] += 10000
            print('sum of heat after case:', sum(self.pptDict.values()))
        # self.pptDict = {key: val/sum(self.pptDict.values()) for key, val in self.pptDict.items()}

        for nodeId, nodeProfit in self.pptDict.items():
            for scube in self.nodesList[nodeId].scubeList:
                geoProfit[scube] += nodeProfit / len(self.nodesList[nodeId].scubeList)
        
        geoProfit = np.array(geoProfit) / np.sum(geoProfit)
        geoProfit = geoProfit.tolist()
        # print('geoProfit', geoProfit)
        heatMap = copy.deepcopy(meta.heatMap)
        for idx, heat in enumerate(geoProfit):
            heatMap[idx]['value'] = heat
        return heatMap
    
    def getDRLInput(self):
        DRLInput = [0] * len(self.dataIdList)
        for dataSet in meta.recordData:
            for dataId in dataSet:
                DRLInput[self.dataIdDict[dataId]] = 1
        return DRLInput
    
    def DRLTrain(self, recommendNum=10):
        self.mtcsStep()
        self.pptDict = {}
        for root in self.rootsList:
            self.getPpt(root)
        self.pptDict = {key: val/sum(self.pptDict.values()) for key, val in self.pptDict.items()}

        geoProfit = [0] * (self.scubeNum * meta.sourceNum * meta.conditionTypeNum)
        for nodeId, nodeProfit in self.pptDict.items():
            for scube in self.nodesList[nodeId].scubeList:
                targetIdx = self.getTargetIdx(scube, self.nodesList[nodeId].source, 0)
                geoProfit[targetIdx] += nodeProfit / len(self.nodesList[nodeId].scubeList)
        
        pptSortedList = sorted(self.pptDict.items(), key=lambda item:item[1], reverse=True)
        idxSortedList = [item[0] for item in pptSortedList]
        recommendList = [{'id': idx, 
            'father': self.nodesParent[idx], 
            'source': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].source], 
            'dataid': self.nodesList[idx].dataIdFromFather,
            'mode': self.nodesList[idx].conditionType,
            'sqlobject': self.nodesList[idx].conditionDict}
            for idx in idxSortedList[: recommendNum]]
            
        return {'input': self.getDRLInput(), 'target': geoProfit}, recommendList
    
    def nodesRecommendByDRL(self, targetProfit, recommendNum=3):
        start = datetime.datetime.utcnow()
        # # targetProfit = [0] * (self.scubeNum * meta.sourceNum * meta.conditionTypeNum)
        # targetProfit = np.random.rand(self.scubeNum * meta.sourceNum * meta.conditionTypeNum)
        # targetProfit = targetProfit / np.sum(targetProfit)
        # targetProfit = targetProfit.tolist()
        # # print(len(targetProfit), sum(targetProfit))
        
        def recordAvailChild(current):
            cnode = self.nodesList[current]
            for idx, child in enumerate(cnode.possible_children):
                timeDetail = []
                if cnode.children_indices[idx] < 0:
                    resultId, conditionType, source = child
                    cresult = cnode.result[resultId]
                    # record infomation of all available children
                    if cresult['id'] not in self.availChildDict.keys():
                        self.availChildDict[cresult['id']] = {
                            'father': current, 
                            'source': ["people", "car", "blog", "point_of_interest"][source], 
                            'dataid': cresult['id'],
                            'mode': conditionType,
                            'sqlobject': cresult['bbx'],
                            'scubeList': cresult['scube'],
                            'stcubeList': cresult['stcubes']
                        }
                    for scube in cresult['scube']:
                        targetIdx = self.getTargetIdx(scube, source, conditionType)
                        if targetIdx not in self.target2dataId.keys():
                            self.target2dataId[targetIdx] = [cresult['id']]
                        elif cresult['id'] not in self.target2dataId[targetIdx]:
                            self.target2dataId[targetIdx].append(cresult['id'])
            for targetIdx, dataIdList in self.target2dataId.items():
                for dataId in dataIdList:
                    profitPart = targetProfit[targetIdx] / len(dataIdList)
                    if dataId not in self.dataId2profit.keys():
                        self.dataId2profit[dataId] = profitPart
                    else:
                        self.dataId2profit[dataId] += profitPart
            # traverse all children nodes
            childIdx = self.nodesChildren[current]
            if len(childIdx) != 0:
                print('traverse all children nodes')
                for idx in childIdx:
                    recordAvailChild(idx)
        
        for root in self.rootsList:
            recordAvailChild(root)
        
        end1 = datetime.datetime.utcnow()
        print('Time 1 consume:', end1-start)
        
        dataIdSortedList = sorted(self.dataId2profit.items(), key=lambda item:item[1], reverse=True)
        dataIdSortedList = [item[0] for item in dataIdSortedList][: recommendNum]
        meta.copyFromRecord()
        for dataId in dataIdSortedList:
            dataInfo = self.availChildDict[dataId]
            cid = self.constructNewNodefromChild(dataInfo['father'], {
                'source': dataInfo['source'], 
                'dataIdFromFather': dataId,
                'sourceFromFather': self.nodesList[dataInfo['father']].source,
                'scubeList': dataInfo['scubeList'],
                'conditionDict': dataInfo['sqlobject'], 
            })
            self.availChildDict[dataId]['id'] = cid
            # 计算每个推荐项的查询结果的stcube的set，总共3个
            # 所有已经获得的数据的stcube的set（所有用户已经建立节点的result）
            # 用上面的3个set分别与下面的set求交集
            # 取出这个交集的大小，用来对3个推荐项排序
            self.dataId2coin[dataId] = len(meta.recordStcubes.union(dataInfo['stcubeList']))
        dataIdSortedByCoin = sorted(self.dataId2coin.items(), key=lambda item:item[1], reverse=True)
        dataIdSortedByCoin = [item[0] for item in dataIdSortedByCoin]
        recommendList = [self.availChildDict[dataId] for dataId in dataIdSortedByCoin]
        for recommend in recommendList:
            recommend.pop('scubeList')
            recommend.pop('stcubeList')
        
        end2 = datetime.datetime.utcnow()
        print('Time 2 consume:', end2-end1)
        return recommendList

        # self.pptDict = {}
        # for root in self.rootsList:
        #     self.getPpt(root)
        
        # targetIdDic = {}            # target id -> node id list
        # for idx in self.pptDict.keys():
        #     for scube in self.nodesList[idx].scubeList:
        #         targetIdx = self.getTargetIdx(
        #             scube, self.nodesList[idx].source, self.nodesList[idx].conditionType)
        #         if targetIdx not in targetIdDic:
        #             targetIdDic[targetIdx] = [idx]
        #         elif idx not in targetIdDic[targetIdx]:
        #             targetIdDic[targetIdx] += [idx]
        
        # idProfitDict = {}           # node id -> profit
        # for targetIdx, idList in targetIdDic.items():
        #     for nodeId in idList:
        #         if nodeId in idProfitDict:
        #             idProfitDict[nodeId] += targetProfit[targetIdx] / len(idList)
        #         else:
        #             idProfitDict[nodeId] = targetProfit[targetIdx] / len(idList)

        # profitSortedList = sorted(idProfitDict.items(), key=lambda item:item[1], reverse=True)
        # idxSortedList = [item[0] for item in profitSortedList]
        # recommendList = [{'id': idx, 
        #     'father': self.nodesParent[idx], 
        #     'source': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].source], 
        #     'dataid': self.nodesList[idx].dataIdFromFather,
        #     'mode': self.nodesList[idx].conditionType,
        #     'sqlobject': self.nodesList[idx].conditionDict}
        #     for idx in idxSortedList[: recommendNum]]
        # return recommendList

    def nodesRecommend(self):
        # startT = datetime.datetime.utcnow()
        self.mtcsStep()
        self.pptDict = {}
        for root in self.rootsList:
            self.getPpt(root)
        pptSortedList = sorted(self.pptDict.items(), key=lambda item:item[1], reverse=True)
        idxSortedList = [item[0] for item in pptSortedList]
        recommendList = [{'id': idx, 
            'father': self.nodesParent[idx], 
            'source': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].source], 
            'sourceFromFather': ["people", "car", "blog", "point_of_interest"][self.nodesList[idx].sourceFromFather], 
            'resultLen': self.nodesList[idx].resultLen,
            'dataid': self.nodesList[idx].dataIdFromFather,
            'mode': self.nodesList[idx].conditionType,
            'sqlobject': self.nodesList[idx].conditionDict}
            for idx in idxSortedList]
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
        print('max depth of tree:', max(self.depthList))
        # height normalization
        normHeightList = self.heightList / (max(np.max(self.depthList), self.depthL) + 1)
        pptList = [node.ppt for node in self.nodesList]
        importList = [(1 - normHeightList[i]) * pptList[i] * self.currentNodesFlag[i] for i in range(len(pptList))]
        importList = np.array(importList) / (np.sum(importList))
        # print('importList:', importList)
        # sample node based on importance of nodes
        sampleIdx = np.random.choice(list(range(nodeNum)), 1, p=importList)[0]
        # print('sampleIdx:', sampleIdx)
        return sampleIdx, self.depthList[sampleIdx]
    
    def getConditionDict(self, conditionType, fullCondition):
        conditionDict = {}
        conditionTypeList = {0: [1, 1], 1: [1, 0], 2: [0, 1]}[conditionType]
        for idx, val in enumerate(conditionTypeList):
            if val == 1:
                conditionType = ['time', 'geo', 'TBD'][idx]
                conditionDict[conditionType] = fullCondition[conditionType]
        return conditionDict

    def selectNode(self, croot):
        current = croot
        childrenIndices = self.nodesList[current].children_indices
        if len(childrenIndices) == 0: return None

        getScore = lambda fatherId, childId: self.nodesList[childId].ppt + 2 * math.sqrt(
            math.log(self.nodesList[fatherId].times, math.e) / self.nodesList[childId].times)
        
        while -1 not in childrenIndices:
            maxScore = 0
            for childIdx in childrenIndices:
                score = getScore(current, childIdx)
                if score > maxScore:
                    maxScore = score
                    nextFather = childIdx 
            current = nextFather
            childrenIndices = self.nodesList[current].children_indices
        # 从当前的父节点中选择一个子节点
        currentFather = self.nodesList[current]
        selectList = [idx for idx in range(len(childrenIndices)) if childrenIndices[idx] == -1]
        selectChildIdx = choice(selectList)

        resultIdx, conditionType, selectSource = currentFather.possible_children[selectChildIdx]
        conditionDict = self.getConditionDict(
            conditionType, currentFather.result[resultIdx]['bbx'])
        expandIndice = self.constructNewNodefromChild(current, child={'source': selectSource,
            'dataIdFromFather': currentFather.result[resultIdx]['id'],
            'sourceFromFather': currentFather.source,
            'scubeList': currentFather.result[resultIdx]['scube'],
            'conditionDict': conditionDict})
        
        currentFather.children_indices[selectChildIdx] = expandIndice
        return expandIndice

    def randomSelectChild(self, node):
        selectList = [idx for idx in range(len(node.children_indices))
            if node.children_indices[idx] == -1]
        if len(selectList) == 0:
            return None
        selectIdx = choice(selectList)
        dataIdx, conditionType, source = node.possible_children[selectIdx]
        conditionDict = self.getConditionDict(
            conditionType, node.result[dataIdx]['bbx'])
        selectNode = queryNode(source=source,
                            dataIdFromFather=node.result[dataIdx]['id'],
                            sourceFromFather=node.source,
                            scubeList=node.result[dataIdx]['scube'],
                            queryFrom='sys',
                            conditionDict=conditionDict,
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
            if cnode is None:
                break
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
            # self.nodesList[current].profit = cprofit
            # self.nodesList[current].profitHistory.append(cprofit)
            # self.nodesList[current].ppt = np.mean(self.nodesList[current].profitHistory)
            self.nodesList[current].profit += cprofit
            self.nodesList[current].times += 1
            self.nodesList[current].ppt = self.nodesList[current].profit / self.nodesList[current].times
            current = self.nodesParent[current]
    
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