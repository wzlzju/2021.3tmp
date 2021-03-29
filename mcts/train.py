from flask import Flask, request, Response
from flask_cors import cross_origin
import numpy as np

app = Flask(__name__)

from MCTS import metadata, queryNode, mcts
from solver import Solver
from query import queryObj
from api import API
from tools import *

class DRLTrainer(object):
    def __init__(self):
        self.m = mcts(queryObj())
        self.trainEpoch = 10
        self.testEpoch = 5
        self.testStep = 5
        self.solver = Solver()
        self.solver.load_model('../checkpoint/tmp.pth')
    
    def train(self):
        for i in range(self.trainEpoch):
            # response = API().query(host='host ip', url='url', type='get')
            pass
        
        # test
        for i in range(self.testEpoch):
            queryCondition = API().query(type='get', url='py/mockReq')
            print(queryCondition)
            self.m.initialization()
            self.m.constructNewNodefromCondition(queryCondition['attr'], queryCondition['source'])
            profQList = []
            for _ in range(self.testStep):
                # payload = {'input': self.m.getDRLInput()}
                solverInput = np.array([self.m.getDRLInput()])
                # print(np.shape(solverInput))
                # vDistrib = API().query(host='host ip', url='url', type='post', payload=payload)
                vDistrib = self.solver.test(solverInput)
                recommendList = self.m.nodesRecommendByDRL(vDistrib, recommendNum=1)
                choiceId = recommendList[0]['id']
                self.m.confirmNode(choiceId)
                profQList.append(self.m.nodesList[choiceId].profQ)
            print('profQList:', profQList)


if __name__ == "__main__":
    trainer = DRLTrainer()
    trainer.train()