from flask import Flask, request, Response
from flask_cors import cross_origin

app = Flask(__name__)

from MCTS import mcts, metadata
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
mBeta = mcts(query.queryObj(), timeL=10)
# Case 1
stepId2nodeId = {0: 0, 1: 0, 2: 2, 3: 2}
# Case 2
# stepId2nodeId = {0: 0, 1: 0, 2:2, 3: 2}
needHeatMap = {0: False, 1: True, 2: False, 3: True}


@app.route("/recommend", methods=["GET", "POST"])
@cross_origin()
def recommend():
    sourceDict = {"people": 0, "car": 1, "blog": 2, "point_of_interest": 3}
    sourceDictForCase = {"people": 0, "taxi": 1, "weibo": 2, "point_of_interest": 3}
    # conditionTypeDict = {"time": "T", "geo": "S"}

    # data = {
    #     "behavior": "rootQuery",
    #     "source": "car",
    #     "sqlobject": {"time": ["00:00:00", "05:00:00"], "geo": [130, 110, 30, 20]},
    # }
    # data = {
    #     "behavior": "childQuery",
    #     "source": "car",
    #     "father": 0,
    #     "dataid": "001a7d352bbe17d45bf6be3b",
    #     "sqlobject": {"time": ["00:00:00", "05:00:00"], "geo": [130, 110, 30, 20]},
    #     "datasource": "point_of_interest",
    # }
    # returnResult = {
    #     "id": 1,
    #     "recommend": [
    #         {
    #             "id": 3,
    #             "father": 1,
    #             "source": "people",
    #             "type": "S",
    #             "data": [
    #                 120.66808428955079,
    #                 120.66408428955079,
    #                 28.01710810852051,
    #                 28.01310810852051,
    #             ],
    #         },
    #         {
    #             "id": 7,
    #             "father": 1,
    #             "source": "car",
    #             "type": "T",
    #             "data": ["00:03:00", "00:07:00"],
    #         },
    #     ],
    # }

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
                'time': ["00:00:00", "08:00:00"]
            }
        print('sqlObject', sqlObject)
        # rootNode = m.constructNewNodefromCondition(sqlObject, source, stepId)
        rootNode = m.constructNewNodefromCondition(sqlObject, source)
        print('Children number of root:', len(m.nodesList[rootNode].possible_children))

        if data.get('case') is not None:
            case = data.get('case')
            case['source'] = sourceDictForCase[case.get("source")]
            # caseNodeId = m.constructNewNodefromCondition(case['sqlobject'], case['source'])
            # caseNodeId = m.constructNewNodefromQuery(stepId2nodeId[case['step']], case['dataid'], case['sqlobject'], case['source'])
            nodeId = stepId2nodeId[case['step']]
            caseNodeId = m.constructNewNodefromChild(nodeId, {
                'source': case['source'],
                'dataIdFromFather': case['dataid'],
                'sourceFromFather': m.nodesList[nodeId].source,
                'scubeList': [1],
                'conditionDict': case['sqlobject']
            })

        mBeta = copy.deepcopy(m)
        recommendList = mBeta.nodesRecommend(3)

        if data.get('case') is not None:
            recommendList = [{
                'id': caseNodeId, 
                'father': m.nodesParent[caseNodeId], 
                'source': ["people", "car", "blog", "point_of_interest"][m.nodesList[caseNodeId].source], 
                'sourceFromFather': ["people", "car", "blog", "point_of_interest"][m.nodesList[caseNodeId].sourceFromFather], 
                'resultLen': m.nodesList[caseNodeId].resultLen,
                'dataid': m.nodesList[caseNodeId].dataIdFromFather,
                'mode': m.nodesList[caseNodeId].conditionType,
                'sqlobject': m.nodesList[caseNodeId].conditionDict,
                'iscase': True
            }] + recommendList[: 2]

        returnResult = {'id': rootNode, 'recommend': recommendList}
        if needHeatMap[case['step']] is True:
            returnResult['heatmap'] = m.drawHeatMap()
    
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
                'time': ["07:00:00", "08:00:00"]
            }
        childNode = m.constructNewNodefromQuery(
            father, dataId, sqlObject, source
        )
        returnResult = {'id': childNode, 'recommend': m.nodesRecommendByDRL()}
    
    elif behavior == "selectRecommend":
        cidx = data.get("id")
        m.confirmNode(cidx)

        if data.get('case') is not None:
            case = data.get('case')
            case['source'] = sourceDictForCase[case.get("source")]
            nodeId = stepId2nodeId[case['step']]
            print('case[source]:', case['source'])
            caseNodeId = m.constructNewNodefromChild(nodeId, {
                'source': case['source'],
                'dataIdFromFather': case['dataid'],
                'sourceFromFather': m.nodesList[nodeId].source,
                'scubeList': [1],
                'conditionDict': case['sqlobject']
            })
        
        mBeta = copy.deepcopy(m)
        recommendList = mBeta.nodesRecommend(3)

        if data.get('case') is not None:
            recommendList = [{
                'id': caseNodeId, 
                'father': m.nodesParent[caseNodeId], 
                'source': ["people", "car", "blog", "point_of_interest"][m.nodesList[caseNodeId].source], 
                'sourceFromFather': ["people", "car", "blog", "point_of_interest"][m.nodesList[caseNodeId].sourceFromFather], 
                'resultLen': m.nodesList[caseNodeId].resultLen,
                'dataid': m.nodesList[caseNodeId].dataIdFromFather,
                'mode': m.nodesList[caseNodeId].conditionType,
                'sqlobject': m.nodesList[caseNodeId].conditionDict, 
                'iscase': True
            }] + recommendList[: 2]
        returnResult = {"recommend": recommendList}
    
    else:
        returnResult = {}
    
    for i in range(len(m.nodesList)):
        m.nodesList[i].profit = mBeta.nodesList[i].profit * meta.decay
        m.nodesList[i].times = mBeta.nodesList[i].times * meta.decay
        m.nodesList[i].ppt = mBeta.nodesList[i].ppt

    print(m.nodesParent)
    print('returnResult:', returnResult)
    response = Response(json.dumps(returnResult), mimetype="application/json")
    # response.headers["Access-Control-Allow-Origin"] = "*"
    # print(response.headers)
    return response

@app.route("/init", methods=["GET", "POST"])
@cross_origin()
def nodeInit():
    m.initialization()
    response = Response(json.dumps('init OK'), mimetype="application/json")
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
