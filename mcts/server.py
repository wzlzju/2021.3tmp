from flask import Flask, request, Response
from flask_cors import cross_origin

app = Flask(__name__)

from MCTS import mcts, metadata
from solver import Solver
import numpy as np
import query
import json
import copy


def requestParse(req_data):
    """解析请求数据并以json形式返回"""
    if req_data.method == "POST":
        data = req_data.json
    elif req_data.method == "GET":
        data = req_data.args
    return data


meta = metadata()
m = mcts(query.queryObj(), timeL=10)
solver = Solver()
solver.load_model('../checkpoint/tmp.pth')


@app.route("/recommend", methods=["GET", "POST"])
@cross_origin()
def recommend():
    sourceDict = {"people": 0, "car": 1, "blog": 2, "point_of_interest": 3}

    data = requestParse(request)
    print('data:', data)
    behavior = data.get("behavior")
    if behavior == "rootQuery":
        source = sourceDict[data.get("source")]
        sqlObject = data.get("sqlobject")
        if sqlObject.get('geo') is None and sqlObject.get('time') is None:
            sqlObject = {
                # 'geo': [120.69783926010132, 120.61760859012602, 28.013147821134936, 27.012896816979197],
                'geo': [120.89783926010132, 120.39760859012602, 28.13147821134936, 27.012896816979197],
                'time': ["00:00:00", "00:10:00"]
            }
        print('sqlObject', sqlObject)
        # rootNode = m.constructNewNodefromCondition(sqlObject, source, stepId)
        rootNode = m.constructNewNodefromCondition(sqlObject, source)
        print('Children number of root:', len(m.nodesList[rootNode].possible_children))

        DRLInput = m.getDRLInput()
        print('DRLInput[: 100]', DRLInput[: 100])
        targetProfit = solver.test(np.array([DRLInput]))
        # print('targetProfit', np.shape(targetProfit), targetProfit)
        recommendList = m.nodesRecommendByDRL(targetProfit=targetProfit)
        
        heatMap = copy.deepcopy(meta.heatMap)
        for idx, heat in enumerate(targetProfit):
            scudeIdx = idx // (meta.conditionTypeNum * meta.sourceNum)
            heatMap[scudeIdx]['value'] += heat * 2

        returnResult = {'id': rootNode, 'recommend': recommendList[: 3], 'heatmap': heatMap}

    elif behavior == "childQuery":
        source = sourceDict[data.get("source")]
        father, dataId, sqlObject = (
            data.get("father"),
            data.get("dataid"),
            data.get("sqlobject"),
        )
        if sqlObject.get('geo') is None and sqlObject.get('time') is None:
            sqlObject = {
                # 'geo': [120.69783926010132, 120.61760859012602, 28.013147821134936, 27.012896816979197],
                'geo': [120.89783926010132, 120.39760859012602, 28.13147821134936, 27.012896816979197],
                'time': ["00:00:00", "08:00:00"]
            }
        childNode = m.constructNewNodefromQuery(
            father, dataId, sqlObject, source
        )
        DRLInput = m.getDRLInput()
        returnResult = {'id': childNode, 'recommend': m.nodesRecommendByDRL(targetProfit=DRLInput)}
    
    elif behavior == "selectRecommend":
        cidx = data.get("id")
        m.confirmNode(cidx)

        DRLInput = m.getDRLInput()
        targetProfit = solver.test(np.array([DRLInput]))
        # print('targetProfit', targetProfit)
        recommendList = m.nodesRecommendByDRL(targetProfit=targetProfit)
        
        heatMap = copy.deepcopy(meta.heatMap)
        for idx, heat in enumerate(targetProfit):
            scudeIdx = idx // (meta.conditionTypeNum * meta.sourceNum)
            heatMap[scudeIdx]['value'] += heat * 2
        
        returnResult = {'recommend': recommendList[: 3], 'heatmap': heatMap}

        
        # DRLInput = m.getDRLInput()
        # recommendList = m.nodesRecommendByDRL(targetProfit=DRLInput)
        # returnResult = {"recommend": recommendList[: 3]}

    else:
        returnResult = {}

    print('returnResult:', returnResult)
    response = Response(json.dumps(returnResult), mimetype="application/json")
    return response

@app.route("/init", methods=["GET", "POST"])
@cross_origin()
def nodeInit():
    m.initialization()
    response = Response(json.dumps('init OK'), mimetype="application/json")
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
