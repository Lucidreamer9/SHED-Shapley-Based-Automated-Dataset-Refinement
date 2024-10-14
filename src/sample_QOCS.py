import json
import sys
def main():
    clusternumber=int(sys.argv[1]) # Number of clusters
    cluster_sequence=[] # Store cluster center data for locating corresponding clusters
    f_1=open("./workspace/cluster_center_"+str(clusternumber)+".json")# Cluster center file, you should use the txt_json.py to convert the txt file to json file
    jsonObect_1=json.load(f_1)
    for i in jsonObect_1:
        cluster_sequence.append(i)

    thecenter=[] #store cluster center instance
    f=open("./workspace/s"+str(clusternumber)+".json")# Score file
    jsonObect=json.load(f)
    for i in jsonObect:
        thecenter.append(i)

    index_map = {item['input']: index for index, item in enumerate(cluster_sequence)}

    # Iterate over list B to find corresponding 'input' index in list A
    cluster_order = [index_map.get(item['input'], None) for item in thecenter]
    print(cluster_order)



  
    finalsetnumber=int(sys.argv[2])
        # Number of data points to extract for the final dataset
    finalset=[]
    for i in cluster_order:
        with open("./workspace/cluster_"+str(clusternumber)+"_"+str(i)+".txt", "r+", encoding="utf-8") as ori:# Cluster file
            for item in ori:
                finalset.append(eval(item))
                if len(finalset)>finalsetnumber-1:
                    print(cluster_order.index(i))    # Print the index of the cluster
                    print(thecenter[cluster_order.index(i)])
                    break
        if len(finalset)>finalsetnumber-1:
            break

        json_temp=open("./final_dataset/"+str(clusternumber)+"/qocs"+str(finalsetnumber)+".json", "w", encoding="utf-8")# Output file
        h=json.dumps(finalset,indent=1)
        json_temp.write(h)


if __name__ == "__main__":
    main()