import requests
import json
import time

#  refer https://cloud.tencent.com/developer/article/1624558
class RequestHandler:
    def get(self, url, **kwargs):
        """封装get方法"""
        # 获取请求参数
        params = kwargs.get("params")
        # headers = kwargs.get("headers")
        headers = {"Accept": "application/json"}
        try:
            result = requests.get(url, params=params, headers=headers)
            return result
        except Exception as e:
            print("get请求错误: %s" % e)

    def post(self, url, **kwargs):
        """封装post方法"""
        # 获取请求参数
        params = kwargs.get("params")
        data = kwargs.get("data")
        json = kwargs.get("json")
        try:
            result = requests.post(url, params=params, data=data, json=json)
            return result
        except Exception as e:
            print("post请求错误: %s" % e)


def convert_resbody_to_obj(res):
    if res == None or res.status_code != 200:
        raise Exception("接口 " + str(res.status_code))
    res = res.content.decode("utf8")
    data = json.loads(res)
    return data


HOST = "http://10.76.0.196:7001/"


class API:
    def getTest(self):
        res = RequestHandler().get(HOST + "test")
        data = convert_resbody_to_obj(res)
        print("GET Test", data)

    def postTest(self, **kwargs):
        payload = kwargs.get("payload")
        res = RequestHandler().post(HOST + "test", json=payload)
        data = convert_resbody_to_obj(res)
        print("POST Test", data)

    def taxi(self, **kwargs):
        payload = kwargs.get("payload")
        res = RequestHandler().post(HOST + "taxi", json=payload)
        data = convert_resbody_to_obj(res)
        print("Taxi Data: ", len(data))

    # def query(self, **kwargs):
    #     payload = kwargs.get("payload")
    #     # print('payload:', payload)
    #     res = RequestHandler().post(HOST + "py/query", json=payload)
    #     data = convert_resbody_to_obj(res)
    #     # data = [
    #     #     {
    #     #         "id": "浙C04937.LOG",
    #     #         "stcubes": [23, 32, 33],
    #     #         "bbx": {
    #     #             "timeRange": ["00:00:03", "23:59:45"],
    #     #             "areaRange": [120.723029, 120.627524, 28.027669, 27.988246],
    #     #         },
    #     #     },
    #     # ]
    #     # convert
    #     # print("Query Data: ", len(data))
    #     return data
    
    def queryByDataId(self, **kwargs):
        payload = kwargs.get("payload")
        res = RequestHandler().post(HOST + "py/queryByDataId", json=payload)
        data = convert_resbody_to_obj(res)
        return data
    
    def query(self, **kwargs):
        requestType = kwargs.get("type")
        url = kwargs.get("url")
        if requestType is 'get':
            res = RequestHandler().get(HOST+url)
        else:
            payload = kwargs.get("payload")
            res = RequestHandler().post(HOST+url, json=payload)
        data = convert_resbody_to_obj(res)
        return data


if __name__ == "__main__":
    # 以下是测试代码
    # get请求接口
    # API().getTest()
    # post请求接口
    # payload = {"username": "vivi", "password": "123456"}
    # API().postTest(payload=payload)

    payload = {
        "geo": [120.707524, 120.623029, 28.027669, 27.988246],
        "time": ["00:06:33", "03:12:56"],
    }
    API().query(payload=payload, url="py/query", type="post")
