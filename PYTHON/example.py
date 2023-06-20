import os
import sys
arg1 = sys.argv[1] #path videoS
videos=os.listdir(arg1)
for video in videos:
    print(video)