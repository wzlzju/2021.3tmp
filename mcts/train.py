from flask import Flask, request, Response
from flask_cors import cross_origin
import numpy as np
import json

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
        self.testStep = 1
        self.solver = Solver()
        self.solver.load_model('../checkpoint/tmp.pth')
    
    def train(self):
        for i in range(self.trainEpoch):
            # response = API().query(host='host ip', url='url', type='get')
            pass
        
        # test
        conditionList = []
        with open('data/test.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                queryCondition = json.loads(line)
                conditionList.append(queryCondition['condition'])
        self.testEpoch = len(conditionList)

        profQList = []
        for i in range(self.testEpoch):
            print('Epoch:', i)
            queryCondition = conditionList[i]
            # print(queryCondition)
            self.m.initialization()
            self.m.constructNewNodefromCondition(queryCondition['attr'], queryCondition['source'])
            
            for _ in range(self.testStep):
                # payload = {'input': self.m.getDRLInput()}
                solverInput = np.array([self.m.getDRLInput()])
                # print(np.shape(solverInput))
                # vDistrib = API().query(host='host ip', url='url', type='post', payload=payload)
                vDistrib = self.solver.test(solverInput)
                # print(np.shape(vDistrib))
                # vDistrib = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.] * 50
                recommendList = self.m.nodesRecommendByDRL(vDistrib, recommendNum=1)
                choiceId = recommendList[0]['id']
                self.m.confirmNode(choiceId)
                profQList.append(self.m.nodesList[choiceId].profQ)
            print('profQList:', profQList)


if __name__ == "__main__":
    trainer = DRLTrainer()
    trainer.train()