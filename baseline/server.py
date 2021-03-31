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
# Case 1
# stepId2nodeId = {0: 0, 1: 0, 2: 2, 3: 2}
# needHeatMap = {0: False, 1: False, 2: False, 3: False, 4: False}
# Case 2
# stepId2nodeId = {0: None, 1: 1, 2: None, 3: 1, 4: None, 5: None}
# needHeatMap = {0: False, 1: True, 2: False, 3: True, 4: True}


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
                'time': ["00:00:00", "00:10:00"]
            }
        print('sqlObject', sqlObject)
        # rootNode = m.constructNewNodefromCondition(sqlObject, source, stepId)
        rootNode = m.constructNewNodefromCondition(sqlObject, source)
        print('Children number of root:', len(m.nodesList[rootNode].possible_children))

        recommendList = m.nodesRecommend()
        # recommendList = [item for item in recommendList if item['source'] != 'point_of_interest']
        returnResult = {'id': rootNode, 'recommend': recommendList[: 3]}
    
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
        returnResult = {'id': childNode, 'recommend': m.nodesRecommendByDRL()}
    
    elif behavior == "selectRecommend":
        cidx = data.get("id")
        m.confirmNode(cidx)
        
        recommendList = m.nodesRecommend()
        # recommendList = [item for item in recommendList if item['source'] != 'point_of_interest']
        returnResult = {"recommend": recommendList[: 3]}
    
    else:
        returnResult = {}

    # print(m.nodesParent)
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
