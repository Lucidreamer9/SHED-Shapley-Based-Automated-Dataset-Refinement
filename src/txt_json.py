import json


obj_temp=[]
with open("/path/to/cluster_center.txt", "r+", encoding="utf-8") as ori:
    item: str
    for item in ori:
        obj_temp.append(eval(item))

json_temp=open("/path/to/cluster_center.json", "w", encoding="utf-8")
h=json.dumps(obj_temp,indent=1)
json_temp.write(h)
