from flask import Flask, request, Response
from flask_cors import cross_origin

app = Flask(__name__)

from MCTS import metadata, queryNode, mcts
from query import queryObj
from api import API
from tools import *

class DRLTrainer(object):
    def __init__(self):
        self.m = mcts(queryObj())
        self.trainEpoch = 10
        self.testEpoch = 5
        self.testStep = 5
    
    def train(self):
        for i in range(self.trainEpoch):
            response = API().query(host='host ip', url='url', type='get')
        
        # test
        for i in range(self.testEpoch):
            queryCondition = API().query(type='get', url='py/mockReq')
            print(queryCondition)
            self.m.initialization()
            self.m.constructNewNodefromCondition(queryCondition['attr'], queryCondition['source'])
            profQList = []
            for _ in range(self.testStep):
                payload = {'input': self.m.getDRLInput()}
                vDistrib = API().query(host='host ip', url='url', type='post', payload=payload)
                recommendList = self.m.nodesRecommendByDRL(vDistrib, recommendNum=1)
                self.m.confirmNode(recommendList[0])
                profQList.append(self.m.nodesList[recommendList[0]].profQ)
            print('profQList:', profQList)

