import json
import ast
import itertools
from ast import literal_eval
import sys

 

def march_instance_score(instance_path,acc_path,base):
    instance_score=[] # Store the score for each instance in one iteration
    acc=[] # Store accuracy
    score=[] # Store score
    # Read accuracy after each iteration
    # with open(acc_path, "r+", encoding="utf-8") as ori:
    #     lines=ori.readlines()
    #     for line in lines:
    #         acc.append(float(line[54:70]))# Modify the index according to the actual situation in the count file. make sure to get the correct accuracy value
    with open(acc_path, "r+", encoding="utf-8") as ori:
        lines = ori.readlines()
        for line in lines:
            data = literal_eval(line.strip())  # Safely evaluate the string as a Python literal
            acc.append(float(data['accuracy']))  # Append the accuracy value to your list

    for i in range(1,50): # Number of accuracy values
        score.append(acc[i-1]-acc[i])
    score.append(acc[49]-base)

    


    # Read instances from randomout
    f=open(instance_path)
    instance=json.load(f)





    for i in range(50):  # Number of accuracy values
        for j in range(60):  # Modify incremental number as needed
            instance[(i*60+j)]['score']=score[i]
            instance_score.append(instance[(i*60+j)])
    return instance_score


def main():
    instance_number=int(sys.argv[1]) # Instance number

    combined_range=range(1,20) # Modify the range as needed

    list_merge=[]
    for i in combined_range:# Iteration number, randomout file, accuracy file, base accuracy
        list_temp=march_instance_score("./workspace/randomout_"+str(instance_number)+"_"+str(i)+".json","./workspace/count_file_"+str(instance_number)+"_"+str(i)+".txt",0) #0 is the base accuracy, modify it as needed
        list_merge=list_merge+list_temp



    score_sum_dict = {}

    # Iterate over each dictionary in the list
    for data_list in [list_merge]:
        for data_dict in data_list:
            input_key = data_dict.get('instruction')  # Get the value of the input key or instruction, modify it according to the actual situation
            score_value = data_dict.get('score')  # Get the value of the score key
            
            if input_key and score_value:
                if input_key in score_sum_dict:
                    score_sum_dict[input_key] += score_value
                else:
                    score_sum_dict[input_key] = score_value

    # Convert the dictionary to a list of dictionaries
    result_list = [{'input': key, 'score_sum': value} for key, value in score_sum_dict.items()]

    sorted_data = sorted(result_list, key=lambda x: x['score_sum'],reverse=True)
    json_temp=open("/workspace/s"+str(instance_number)+".json", "w", encoding="utf-8")
    h=json.dumps(sorted_data,indent=1)
    json_temp.write(h)

if __name__ == "__main__":
    main()