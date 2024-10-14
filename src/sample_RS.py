import json

import random
clusternumber=3000


for j in range(1,51):
    finalsetnumber=1000*j# Number of data points to extract for the final dataset
    finalset=[]
    tempset=[]
    for i in range(clusternumber):
        with open("path/to/cluster_"+str(clusternumber)+"_"+str(i)+".txt", "r+", encoding="utf-8") as ori:
            for item in ori:
                tempset.append(eval(item))

    finalset.extend(random.choices(tempset,k=finalsetnumber))
    json_temp=open("path/to/sample/"+str(clusternumber)+"/rs"+str(finalsetnumber)+".json", "w", encoding="utf-8")# Output file
    h=json.dumps(finalset,indent=1)
    json_temp.write(h)


