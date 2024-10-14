import json
import math
import random
import sys
def main():
    clusternumber=int(sys.argv[1]) # Number of clusters
    cluster_sequence=[] # Store cluster center data for locating corresponding clusters
    f_1=open("./workspace/cluster_center_"+str(clusternumber)+".json")# Cluster center file, you should use the txt_json.py to convert the txt file to json file
    jsonObect_1=json.load(f_1)
    for i in jsonObect_1:
        cluster_sequence.append(i)

    thecenter=[] # Store the score data of each cluster
    f=open("./workspace/s"+str(clusternumber)+".json")# Score file
    jsonObect=json.load(f)
    for i in jsonObect:
        thecenter.append(i)

    index_map = {item['input']: index for index, item in enumerate(cluster_sequence)}
    cluster_score=[item['score_sum'] for item in thecenter]

    
    cluster_order = [index_map.get(item['input'], None) for item in thecenter]

    def adjust_weight(score):
        return math.exp(score)
    # calculate the adjusted weights
    adjusted_weights=[adjust_weight(score)for score in cluster_score]

    # normalize the adjusted weights
    total_weight = sum(adjusted_weights)
    normalized_weights = [weight / total_weight for weight in adjusted_weights]


   
    finalsetnumber=int(sys.argv[2])


    finalset=[]

    datadict={}
    for i in range(clusternumber):
        with open("./workspace/cluster_"+str(clusternumber)+"_"+str(i)+".txt", "r+", encoding="utf-8") as file:
                data = file.readlines()
        datadict[i]=data
        




    for i in range(finalsetnumber):
        while True:
             
            new_index=random.choices(cluster_order,weights=normalized_weights,k=1)#每条数据对应的cluster的index
            if len(datadict[new_index[0]])<1:
                continue
            selectedline=random.choices(datadict[new_index[0]])
            finalset.append(eval(selectedline[0].strip()))
            datadict[new_index[0]].remove(selectedline[0])
            break
    json_temp=open("./final_dataset/"+str(clusternumber)+"/qwcs"+str(finalsetnumber)+".json", "w", encoding="utf-8")#output_file
    h=json.dumps(finalset,indent=1)
    json_temp.write(h)







    

    


    