import os
import json
import csv
import shapely
from shapely.geometry import Point
from shapely.geometry import Polygon
import time
from api import API

def depth_count(lists,x=0):
    if not lists or not isinstance(lists,list):
        return x
    l = depth_count(lists[0],x + 1)
    r = depth_count(lists[1:],x)
    return max(l,r)

class queryObj(object):
    def __init__(self, path="./data"):
        self.filepath = path
        self.data = {}
        self.readInData()
        self.partition = [100,100,100]  # lng,lat,t
        self.precalculate()
        print("Initialization completed. ")

    def readInData(self):
        with open(self.filepath+"/new_mobile_traj.json", "r") as f:
            self.data["mobileTraj"] = json.load(f)[:1000]      # [[{"time":"2014-01-13 23:58:34.12","lng":float,"lat":float}]]
        with open(self.filepath+"/taxi_421.json", "r") as f:
            self.data["taxiTraj"] = json.load(f)[:1000]        # [[{"time":"2014-01-13 23:58:34.12","lng":float,"lat":float}]]
        with open(self.filepath+'/weibo_2014-01-14_all.csv')as f:
            f_csv = csv.reader(f)
            self.data["weibo"] = [row for row in f_csv][1:][:1000]     # (float:) [[2014-01-14 00:50,120.6462979,28.00516276],...]

    def query(self, source, tRange=None, sRange=None):
        if type(source) is int:
            if source < 0 or source >= len(self.data.keys()):
                print("Unavailable source type")
                return
            source = list(self.data.keys())[source]
        if source not in self.data.keys():
            print("Unavailable source type")
            return
        if tRange is None and sRange is None:
            return self.data[source]
        result = []
        if tRange is None:
            if source == "mobileTraj":
                result = [d for d in self.data["mobileTraj"] if self.lInSRange(sRange,d)]
            elif source == "taxiTraj":
                result = [d for d in self.data["taxiTraj"] if self.lInSRange(sRange,d)]
            elif source == "weibo":
                result = [d for d in self.data["weibo"] if self.pInSRange(sRange,float(d[1]),float(d[2]))]
        elif sRange is None:
            if source == "mobileTraj":
                result = [d for d in self.data["mobileTraj"] if self.lInTRange(tRange,d)]
            elif source == "taxiTraj":
                result = [d for d in self.data["taxiTraj"] if self.lInTRange(tRange,d)]
            elif source == "weibo":
                result = [d for d in self.data["weibo"] if self.pInTRange(tRange,d[0])]
        else:
            if source == "mobileTraj":
                result = [d for d in self.data["mobileTraj"] if self.lInSRange(sRange,d) and self.lInTRange(tRange,d)]
            elif source == "taxiTraj":
                result = [d for d in self.data["taxiTraj"] if self.lInSRange(sRange,d) and self.lInTRange(tRange,d)]
            elif source == "weibo":
                result = [d for d in self.data["weibo"] if self.pInSRange(sRange,float(d[1]),float(d[2])) and self.pInTRange(tRange,d[0])]
        return result

    def queryIdx(self, source, tRange=None, sRange=None):
        if type(source) is int:
            if source < 0 or source >= len(self.data.keys()):
                print("Unavailable source type")
                return
            source = list(self.data.keys())[source]
        if source not in self.data.keys():
            print("Unavailable source type")
            return
        if tRange is None and sRange is None:
            return self.data[source]
        result = []
        if tRange is None:
            if source == "mobileTraj":
                result = [i for i,d in enumerate(self.data["mobileTraj"]) if self.lInSRange(sRange,d)]
            elif source == "taxiTraj":
                result = [i for i,d in enumerate(self.data["taxiTraj"]) if self.lInSRange(sRange,d)]
            elif source == "weibo":
                result = [i for i,d in enumerate(self.data["weibo"]) if self.pInSRange(sRange,float(d[1]),float(d[2]))]
        elif sRange is None:
            if source == "mobileTraj":
                result = [i for i,d in enumerate(self.data["mobileTraj"]) if self.lInTRange(tRange,d)]
            elif source == "taxiTraj":
                result = [i for i,d in enumerate(self.data["taxiTraj"]) if self.lInTRange(tRange,d)]
            elif source == "weibo":
                result = [i for i,d in enumerate(self.data["weibo"]) if self.pInTRange(tRange,d[0])]
        else:
            if source == "mobileTraj":
                result = [i for i,d in enumerate(self.data["mobileTraj"]) if self.lInSRange(sRange,d) and self.lInTRange(tRange,d)]
            elif source == "taxiTraj":
                result = [i for i,d in enumerate(self.data["taxiTraj"]) if self.lInSRange(sRange,d) and self.lInTRange(tRange,d)]
            elif source == "weibo":
                result = [i for i,d in enumerate(self.data["weibo"]) if self.pInSRange(sRange,float(d[1]),float(d[2])) and self.pInTRange(tRange,d[0])]
        return result

    def queryIdxSimplify(self, source, conditionDict):
        # tRange = conditionDict['time'] if 'time' in conditionDict.keys() else None
        # sRange = conditionDict['geo'] if 'geo' in conditionDict.keys() else None
        # if tRange is None and sRange is None:
        #     return self.data[source]
        # result = API().query(payload={'source': source, 'attr': {'time': tRange, 'geo': sRange}})
        if 'time' not in conditionDict and 'geo' not in conditionDict:
            return self.data[source]
        result = API().query(payload={'source': source, 'attr': conditionDict}, 
            url="py/query", type="post")
        return result
    
    def queryByDataId(self, idx, originSource, targetSource, mode):
        if idx is None or originSource is None:
            print('idx:', idx, 'originSource:', originSource)
            exit()
        result = API().queryByDataId(payload={'id': idx, 
            'originSource': originSource, 
            'targetSource': targetSource, 
            'mode': mode})
        return result

    def pInSRange(self, sRange, lng, lat):
        if hasattr(sRange, "geom_type"):
            return Point(lng,lat).within(sRange)
        else:
            dc = depth_count(sRange)
            if dc == 1:    # point
                return Point(lng,lat).almost_equals(Point(sRange),6)
            elif dc == 2:   # normal polygon
                return Point(lng, lat).within(Polygon(sRange))
            elif dc == 3:   # polygon with inner-loop
                return Point(lng, lat).within(Polygon(sRange[0],sRange[1:]))

    def lInSRange(self, sRange, l):
        if not hasattr(sRange, "geom_type"):
            dc = depth_count(sRange)
            if dc == 2:  # normal polygon
                sRange = Polygon(sRange)
            elif dc == 3:  # polygon with inner-loop
                sRange = Polygon(sRange[0], sRange[1:])
        for location in l:
            if self.pInSRange(sRange, location["lng"], location["lat"]):
                return True
        return False

    def pInTRange(self, tRange, t):
        t0 = tRange[0]
        t1 = tRange[1]

        if type(t) is str:
            try:
                t = time.strptime(t.split(".")[0],"%Y-%m-%d %H:%M:%S")
            except:
                try:
                    t = time.strptime(t.split(".")[0], "%Y-%m-%d %H:%M")
                except:
                    print("Abnormal time format:", t)
            t = time.mktime(t)
        else:
            pass

        if isinstance(t0, float):
            pass
        elif isinstance(t0, time.struct_time):
            t0 = time.mktime(t0)
            t1 = time.mktime(t1)
        elif isinstance(t0, str):
            try:
                t0 = time.strptime(t0.split(".")[0],"%Y-%m-%d %H:%M:%S")
            except:
                try:
                    t0 = time.strptime(t0.split(".")[0], "%Y-%m-%d %H:%M")
                except:
                    print("Abnormal time0 format:", t0)
            t0 = time.mktime(t0)
            try:
                t1 = time.strptime(t1.split(".")[0],"%Y-%m-%d %H:%M:%S")
            except:
                try:
                    t1 = time.strptime(t1.split(".")[0], "%Y-%m-%d %H:%M")
                except:
                    print("Abnormal time1 format:", t1)
            t1 = time.mktime(t1)
        else:
            print("Unavailable time format:", type(t0))

        return t>t0 and t<t1 or t<t0 and t>t1


    def lInTRange(self, tRange, l):
        t0 = tRange[0]
        t1 = tRange[1]
        if isinstance(t0, time.struct_time):
            t0 = time.mktime(t0)
            t1 = time.mktime(t1)
        elif isinstance(t0, float):
            pass
        elif isinstance(t0, str):
            try:
                t0 = time.strptime(t0.split(".")[0],"%Y-%m-%d %H:%M:%S")
            except:
                try:
                    t0 = time.strptime(t0.split(".")[0], "%Y-%m-%d %H:%M")
                except:
                    print("Abnormal time0 format:", t0)
            t0 = time.mktime(t0)
            try:
                t1 = time.strptime(t1.split(".")[0],"%Y-%m-%d %H:%M:%S")
            except:
                try:
                    t1 = time.strptime(t1.split(".")[0], "%Y-%m-%d %H:%M")
                except:
                    print("Abnormal time1 format:", t1)
            t1 = time.mktime(t1)
        else:
            print("Unavailable time format:", type(t0))
        for location in l:
            if self.pInTRange([t0,t1], location["time"]):
                return True
        return False

    def precalculate(self):
        llng, rlng, ulat, dlat = 180,-180,-90,90
        if True:
            self.dataFormulate()
        time0, time1 = 1e20, 0
        for t in self.data["mobileTraj"]:
            for p in t:
                if p["lng"] < llng:
                    llng = p["lng"]
                if p["lng"] > rlng:
                    rlng = p["lng"]
                if p["lat"] < dlat:
                    dlat = p["lat"]
                if p["lat"] > ulat:
                    ulat = p["lat"]
                if p["time"] < time0:
                    time0 = p["time"]
                if p["time"] > time1:
                    time1 = p["time"]
        for t in self.data["taxiTraj"]:
            for p in t:
                if p["lng"] < llng:
                    llng = p["lng"]
                if p["lng"] > rlng:
                    rlng = p["lng"]
                if p["lat"] < dlat:
                    dlat = p["lat"]
                if p["lat"] > ulat:
                    ulat = p["lat"]
                if p["time"] < time0:
                    time0 = p["time"]
                if p["time"] > time1:
                    time1 = p["time"]
        for ti,lng,lat in self.data["weibo"]:
            #lng = float(lng)
            #lat = float(lat)
            if lng<llng:
                llng = lng
            if lng>rlng:
                rlng = lng
            if lat<dlat:
                dlat = lat
            if lat>ulat:
                ulat = lat
            if ti < time0:
                time0 = ti
            if ti > time1:
                time1 = ti
        self.bbox = [llng, rlng, ulat, dlat]
        self.tbox = [time0, time1]
        self.brange = [rlng-llng, ulat,dlat]
        self.trange = time1-time0

    def dataFormulate(self):
        for i,t in enumerate(self.data["mobileTraj"]):
            for j,p in enumerate(t):
                ti = p["time"]
                if type(ti) is str:
                    ti = "2014-01-14 "+ti.split(" ")[1]
                    try:
                        ti = time.strptime(ti.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    except:
                        try:
                            ti = time.strptime(ti.split(".")[0], "%Y-%m-%d %H:%M")
                        except:
                            print("Abnormal time format:", ti)
                    ti = time.mktime(ti)
                else:
                    pass
                self.data["mobileTraj"][i][j]["time"] = ti
        for i,t in enumerate(self.data["taxiTraj"]):
            for j,p in enumerate(t):
                ti = p["time"]
                if type(ti) is str:
                    ti = "2014-01-14 "+ti.split(" ")[1]
                    try:
                        ti = time.strptime(ti.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    except:
                        try:
                            ti = time.strptime(ti.split(".")[0], "%Y-%m-%d %H:%M")
                        except:
                            print("Abnormal time format:", ti)
                    ti = time.mktime(ti)
                else:
                    pass
                self.data["taxiTraj"][i][j]["time"] = ti
        for i,[ti,lng,lat] in enumerate(self.data["weibo"]):
            if type(ti) is str:
                ti = "2014-01-14 "+ti.split(" ")[1]
                try:
                    ti = time.strptime(ti.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except:
                    try:
                        ti = time.strptime(ti.split(".")[0], "%Y-%m-%d %H:%M")
                    except:
                        print("Abnormal time format:", ti)
                ti = time.mktime(ti)
            else:
                pass
            self.data["weibo"][i][0] = ti
            self.data["weibo"][i][1] = float(lng)
            self.data["weibo"][i][2] = float(lat)

    def bboxp(self, source, p):
        if source and type(source) is str:
            p = self.data[source][p]
        #lng = float(p[1])
        #lat = float(p[2])
        lng = p[1]
        lat = p[2]
        return [lng-self.brange[0]/self.partition[0]/4, lng+self.brange[0]/self.partition[0]/4,
                lat + self.brange[1] / self.partition[1] / 4, lat - self.brange[1] / self.partition[1] / 4]

    def bboxp2(self, source, ps):
        if len(ps) == 1:
            return self.bboxp(source, ps[0])
        if source and type(source) is str:
            ps = [self.data[source][i] for i in ps]
        llng, rlng, ulat, dlat = 180,-180,-90,90
        for _,lng,lat in ps:
            #lng = float(lng)
            #lat = float(lat)
            lng = lng
            lat = lat
            if lng<llng:
                llng = lng
            if lng>rlng:
                rlng = lng
            if lat<dlat:
                dlat = lat
            if lat>ulat:
                ulat = lat
        return [llng-self.brange[0]/self.partition[0]/4, rlng+self.brange[0]/self.partition[0]/4,
                ulat + self.brange[1] / self.partition[1] / 4, dlat - self.brange[1] / self.partition[1] / 4]

    def bboxt(self, source, t):
        if source and type(source) is str:
            t = self.data[source][t]
        llng, rlng, ulat, dlat = 180,-180,-90,90
        for p in t:
            if p["lng"]<llng:
                llng = p["lng"]
            if p["lng"]>rlng:
                rlng = p["lng"]
            if p["lat"]<dlat:
                dlat = p["lat"]
            if p["lat"]>ulat:
                ulat = p["lat"]
        return [llng - self.brange[0] / self.partition[0] / 4, rlng + self.brange[0] / self.partition[0] / 4,
                ulat + self.brange[1] / self.partition[1] / 4, dlat - self.brange[1] / self.partition[1] / 4]

    def bboxt2(self, source, ts):
        if len(ts) == 1:
            return self.bboxt(source, ts[0])
        if source and type(source) is str:
            ts = [self.data[source][i] for i in ts]
        llng, rlng, ulat, dlat = 180,-180,-90,90
        for t in ts:
            for p in t:
                if p["lng"] < llng:
                    llng = p["lng"]
                if p["lng"] > rlng:
                    rlng = p["lng"]
                if p["lat"] < dlat:
                    dlat = p["lat"]
                if p["lat"] > ulat:
                    ulat = p["lat"]
        return [llng - self.brange[0] / self.partition[0] / 4, rlng + self.brange[0] / self.partition[0] / 4,
                ulat + self.brange[1] / self.partition[1] / 4, dlat - self.brange[1] / self.partition[1] / 4]

    def tboxp(self, source, p):
        if source and type(source) is str:
            p = self.data[source][p]
        ti = p[0]
        return [ti-self.trange/self.partition[2]/4, ti+self.trange/self.partition[2]/4]

    def tboxp2(self, source, ps):
        if len(ps) == 1:
            return self.tboxp(source, ps[0])
        if source and type(source) is str:
            ps = [self.data[source][i] for i in ps]
        time0, time1 = 1e20, 0
        for ti,lng,lat in ps:
            if ti<time0:
                time0 = ti
            if ti>time1:
                time1 = ti
        return [time0-self.trange/self.partition[2]/4, time1+self.trange/self.partition[2]/4]

    def tboxt(self, source, t):
        if source and type(source) is str:
            t = self.data[source][t]
        time0, time1 = 1e20, 0
        for p in t:
            ti = p["time"]
            if ti < time0:
                time0 = ti
            if ti > time1:
                time1 = ti
        return [time0-self.trange/self.partition[2]/4, time1+self.trange/self.partition[2]/4]

    def tboxt2(self, source, ts):
        if len(ts) == 1:
            return self.tboxt(source, ts[0])
        if source and type(source) is str:
            ts = [self.data[source][i] for i in ts]
        time0, time1 = 1e20, 0
        for t in ts:
            for p in t:
                ti = p["time"]
                if ti < time0:
                    time0 = ti
                if ti > time1:
                    time1 = ti
        return [time0-self.trange/self.partition[2]/4, time1+self.trange/self.partition[2]/4]

    def pinbbox(self, source, p, box):
        if source and type(source) is str:
            p = self.data[source][p]
        lng, lat = p[1], p[2]
        l,r,u,d = box[0],box[1],box[2],box[3]
        return lng>=l and lng<=r and lat>=d and lat<=u

    def tinbbox(self, source, t, box):
        if source and type(source) is str:
            t = self.data[source][t]
        l,r,u,d = box[0],box[1],box[2],box[3]
        for p in t:
            lng, lat = p["lng"], p["lat"]
            if lng>=l and lng<=r and lat>=d and lat<=u:
                return True
        return False

    def pintbox(self, source, p, box):
        if source and type(source) is str:
            p = self.data[source][p]
        ti = p[0]
        t0, t1 = box[0],box[1]
        return ti>=t0 and ti<=t1

    def tintbox(self, source, t, box):
        if source and type(source) is str:
            t = self.data[source][t]
        t0, t1 = box[0],box[1]
        for p in t:
            ti = p["time"]
            if ti>=t0 and ti<=t1:
                return True
        return False



if __name__ == "__main__":
    q = queryObj()
    tr = ("2014-01-14 23:50:24.83","2014-01-14 23:54:18.84")
    sr = [[120.6,28],[120.8,27],[120.4,27]]
    print(q.bboxp("weibo", 0))
    print(q.bboxp2("weibo", [1,2,3]))
    print(q.bboxt("mobileTraj", 0))
    print(q.bboxt2("taxiTraj", [0,1,2]))
    print(q.tboxp("weibo", 0))
    print(q.tboxp2("weibo", [1,2,3]))
    print(q.tboxt("mobileTraj", 0))
    print(q.tboxt2("taxiTraj", [0,1,2]))
    print(len(q.queryIdxSimplify("mobileTraj", tRange=[1389631584.005, 1389632015.995])))
    print(len(q.queryIdxSimplify("mobileTraj", sRange=[120.3551055, 120.9374903, 28.13387079, 27.876454730000003])))
    print(len(q.queryIdxSimplify("taxiTraj", tRange=[1389631584.005, 1389632015.995])))
    print(len(q.queryIdxSimplify("taxiTraj", sRange=[120.3551055, 120.9374903, 28.13387079, 27.876454730000003])))
    print(len(q.queryIdxSimplify("weibo", tRange=[1389631584.005, 1389632015.995])))
    print(len(q.queryIdxSimplify("weibo", sRange=[120.3551055, 120.9374903, 28.13387079, 27.876454730000003])))
    #print(len(q.query("mobileTraj", tr, sr)))
    #print(len(q.query("taxiTraj", tr, sr)))
    #print(len(q.query("weibo", tr, sr)))
    for i in range(10000):
        q.queryIdxSimplify("mobileTraj", sRange=[120.3551055, 120.9374903, 28.13387079, 27.876454730000003])
        if(i%1000==0):
            print(i)