import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out_num', type=int, default = None)
parser.add_argument('--out_filepath', type=str, default=None)
parser.add_argument('--temp_save_filepath', type=str, default=None)
args = parser.parse_args()




out_item=[]
f=open(args.out_filepath, "r", encoding="utf-8")  
params = json.load(f)  
i=0
while i<args.out_num and len(params)!=0:
    j=random.randint(0,(len(params)-1))
    temp=params.pop(j)
    out_item.append(temp)
    i=i+1
f.closed   

json_temp=open(args.out_filepath, "w", encoding="utf-8")
h=json.dumps(params,indent=1)
json_temp.write(h)
try:
    file = open(args.temp_save_filepath,'r',encoding="utf-8")
    temp_save=json.load(file)
    temp_save=temp_save+out_item
except json.JSONDecodeError:
    temp_save=out_item
    pass

file_w = open(args.temp_save_filepath,'w',encoding="utf-8")
file_w.write(json.dumps(temp_save,indent=1))

            
        
 
 
 

 

