#example: python evaluation/connectivity.py --model gpt-3.5-turbo-instruct --mode easy --prompt CoT --SC 1 --SC_num 5
#run all the evaluation scripts
import os
import sys
import argparse
import numpy as np
import subprocess
#"connectivity","cycle","shortest_path","flow","hamilton","matching"
if 1==1:
    for subject in ["connectivity","cycle","shortest_path","flow","hamilton","matching"]:
        for mode in ["easy"]:#,"medium","hard"]:
            model="gpt-3.5-turbo-instruct"
            #for prompt in ["CoT",  "0-CoT", "none",  "k-shot"]:
            for prompt in ["skydogli"]:
                cmd="python evaluation/"+subject+".py --model "+model+" --mode "+mode+" --prompt "+prompt+" --SC 0 --SC_num 5 --token 1024"
                print("HI ",cmd)
                os.system(cmd)
                if prompt=="CoT":
                    cmd="python evaluation/"+subject+".py --model "+model+" --mode "+mode+" --prompt "+prompt+" --SC 1 --SC_num 5"
                    os.system(cmd)
if 1==1:
    for subject in ["topology"]:
        for mode in ["easy"]:
            model="gpt-3.5-turbo-instruct"
            #for prompt in ["CoT",  "0-CoT", "none",  "k-shot","LTM"]:
            for prompt in ["skydogli"]:
                cmd="python evaluation/"+subject+".py --model "+model+" --mode "+mode+" --prompt "+prompt+" --SC 0 --SC_num 5"
                print("HI ",cmd)
                os.system(cmd)
                if prompt=="CoT":
                    cmd="python evaluation/"+subject+".py --model "+model+" --mode "+mode+" --prompt "+prompt+" --SC 1 --SC_num 5"
                    os.system(cmd)
