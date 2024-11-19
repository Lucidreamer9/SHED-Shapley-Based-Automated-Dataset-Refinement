import json

num_clusters = int(sys.argv[1])
obj_temp=[]
with open("/workspace/cluster_center_"+str(num_clusters)+".txt", "r+", encoding="utf-8") as ori:
    item: str
    for item in ori:
        obj_temp.append(eval(item))

json_temp=open("/workspace/cluster_center_"+str(num_clusters)+".json", "w", encoding="utf-8")
h=json.dumps(obj_temp,indent=1)
json_temp.write(h)
