import openai
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import time
from datetime import datetime, timedelta, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

model_list = ["text-davinci-003","code-davinci-002","gpt-3.5-turbo","gpt-4"]
parser = argparse.ArgumentParser(description="connectivity")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--mode', type=str, default="easy", help='mode (default: easy)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temprature (default: 0)')
parser.add_argument('--token', type=int, default=256, help='max token (default: 256)')
parser.add_argument('--SC', type=int, default=0, help='self-consistency (default: 0)')
parser.add_argument('--SC_num', type=int, default=5, help='number of cases for SC (default: 5)')
args = parser.parse_args()
assert args.prompt in ["CoT", "none", "0-CoT", "LTM", "PROGRAM","k-shot","Algorithm","Instruct","Repeat","Repeat+CoT","skydogli","skydogli2","rp","rpp"]
def translate(m,q,array,args):
    edge = array[:m]
    question = array[m:]
    Q = ''
    if args.prompt in ["CoT", "k-shot","Algorithm","Instruct","Repeat","Repeat+CoT","skydogli","skydogli2","rp","rpp"]:
        with open("NLGraph/connectivity/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = Q +"Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.\nGraph:"
    for i in range(m):
        Q = Q + ' ('+str(edge[i][0])+','+str(edge[i][1])+')'
    Q = Q + "\n"

    if args.prompt == "Instruct":
        Q = Q + "Let's construct a graph with the nodes and edges first.\n"
    Q = Q + "Q: Is there a path between "
    Q_list = []
    for i in range(q*2):
        Q_i = Q + "node "+str(question[i][0])+" and node "+str(question[i][1])+"?Please express your answer as \"the answer is no\" or \"the answer is yes.\"\nA:"
        match args.prompt:
            case "0-CoT":
                Q_i = Q_i + " Let's think step by step:"
            case "LTM":
                Q_i = Q_i + " Let's break down this problem:"
            case "PROGRAM":
                Q_i = Q_i + " Let's solve the problem by a Python program:"
        Q_list.append(Q_i)
    return Q_list

from openai import OpenAI

client = OpenAI(
    # #将这里换成你在aihubmix api keys拿到的密钥
    api_key="sk-JHYlQsel9VE6RufY0fE0B2EbD0574cF6AaBf5eA623DaF993",
    # 这里将官方的接口访问地址，替换成aihubmix的入口地址
    base_url="https://aihubmix.com/v1"
)
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(1000))
def predict(Q, args):
    input = Q
    temperature = 0
    if args.SC == 1:
        temperature = 0.7
    if 'gpt' in args.model and args.model != "gpt-3.5-turbo-instruct":
        Answer_list = []
        print("Len: ",len(input))
        num=0
        for text in input:
            print("Request ",num)
            num+=1
            try:
                response = client.chat.completions.create(
                model=args.model,
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text},
                ],
                temperature=temperature,
                max_tokens=args.token,
                )
                print(response.choices[0].message.content)
                Answer_list.append(response.choices[0].message.content)
                print("GET RESPONSE")
            except Exception as e:  # 捕获所有类型的异常
                print(f"GG: {e}")
        return Answer_list
    print("GO ",args.token*len(input))
    response = client.completions.create(
    model=args.model,
    prompt=input,
    temperature=temperature,
    max_tokens=args.token*len(input),
    )
    Answer_list = []
    print("Get Response")
    for i in range(len(input)):
#        print("GET RESPONSE: ",response.choices)
        Answer_list.append(response.choices[i].text)
    print("DONE: ",len(Answer_list))
    return Answer_list

def log(Q_list, res, answer, args, existacc, noexistacc):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = 'log/connectivity/'+args.model+'-'+args.mode+'-'+time+'-'+args.prompt
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath+"res.npy", res)
    np.save(newpath+"answer.npy", answer)
    with open(newpath+"prompt.txt","w") as f:
        f.write(Q_list[0])
        f.write("\n")
        f.write("Acc: " + str(res.sum())+'/'+str(len(res)) + '\n')
        print(args, file=f)
    with open(newpath+"analyze.txt","w") as f:
        #write existacc and noexistacc
        f.write("Existacc: "+str(existacc)+"\n")
        f.write("Noexistacc: "+str(noexistacc)+"\n")
def main():
    print("HI")

    if 'OPENAI_API_KEY' in os.environ:
        openai.api_key = os.environ['OPENAI_API_KEY']
        client.api_key = os.environ['OPENAI_API_KEY']
    else:
        raise Exception("Missing openai key!")
    if 'OPENAI_ORGANIZATION' in os.environ:
        openai.organization = os.environ['OPENAI_ORGANIZATION']
    res, answer = [], []
    match args.mode:
        case "easy":
            g_num = 36
        case "medium":
            g_num = 120
        case "hard":
            g_num = 68
    existacc=0.0
    noexistacc=0.0
    for i in tqdm(range(g_num)):
        with open("NLgraph/connectivity/graph/"+args.mode+"/standard/graph"+str(i)+".txt","r") as f:
            n, m ,q = [int(x) for x in next(f).split()]
            array = []
            for line in f: # read rest of lines
                array.append([int(x) for x in line.split()])
            qt = array[m:]
            Q_list = translate(m,q,array, args)
            sc = 1
            if args.SC == 1:
                sc = args.SC_num
            sc_list = []
            for k in range(sc):
                answer_list = predict(Q_list, args)
                sc_list.append(answer_list)
            for j in range(q):
                vote = 0
                for k in range(sc):
                    ans = sc_list[k][j].lower()
                    answer.append(ans)
                    #ans = os.linesep.join([s for s in ans.splitlines() if s]).replace(' ', '')
                    pos1,pos2=ans.rfind("the answer is yes"),ans.rfind("the answer is no")
                    if pos1>pos2:
                        vote += 1
                if vote * 2 >= sc:
                    res.append(1)
                    existacc+=1
                else:
                    res.append(0)
                
            for j in range(q):
                vote = 0
                for k in range(sc):
                    ans = sc_list[k][j+q].lower()
                    answer.append(ans)
                    #ans = os.linesep.join([s for s in ans.splitlines() if s]).replace(' ', '')
                    pos1,pos2=ans.rfind("the answer is yes"),ans.rfind("the answer is no")
                    if pos1<pos2:
                        vote += 1
                if vote * 2 >= sc:
                    res.append(1)
                    noexistacc+=1
                else:
                    res.append(0)

    #write answer to file answer.txt
    with open("answer.txt","w") as f:
        for i in range(len(answer)):
            f.write(answer[i])
            f.write("\n")
    #write res to file res.txt
    with open("res.txt","w") as f:
        for i in range(len(res)):
            f.write(str(res[i]))
            f.write("\n")    
    res = np.array(res)
    answer = np.array(answer)
    existacc = existacc/(len(res)/2)
    noexistacc = noexistacc/(len(res)/2)
    print((res==1).sum())
    log(Q_list, res, answer, args, existacc, noexistacc)
if __name__ == "__main__":
    main()