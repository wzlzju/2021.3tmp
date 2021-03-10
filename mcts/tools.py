import json

def saveJson(infoLog, fileName):
    with open(fileName+'.json', 'a', encoding='utf-8') as f:
        logJson = json.dumps(infoLog, ensure_ascii=False)
        f.write(logJson)
        f.write('\n')