import os
import time

def isVimTempFile(fn):
    # Eg: .screen_play.py.swp 
    return name[0] == '.' and '.sw' in name[-4:-1]

def isTempFile(fn):
    # Eg: .DS_Store 
    return name[0] == '.'

#TopDir='/Users/jiadinggai/dev/LLVM/llvm-project/llvm/lib/Transforms'

# Step 1. print bird's eye view of the directory structure.
fileList = []
for root, dirs, files in os.walk(TopDir, topdown=False):
  for name in files:
    fullPath = os.path.join(root, name)
    if isVimTempFile(name) or isTempFile(name):
        continue

    fileList.append(fullPath)

bs = 30
for file in fileList:
    if bs == 0:
        bs = 30
        time.sleep(0.5)
        print("-------------------------------------")
    print("==> ", file, end="\n")
    bs -= 1
time.sleep(1.2)

# Step 2. print each file in the same order.
for root, dirs, files in os.walk(TopDir, topdown=False):
  for name in files:
    fullPath = os.path.join(root, name)
    if isVimTempFile(name) or isTempFile(name):
        continue

    print('\n' * 10)
    print("=====================================================================================================")
    print("==> ", fullPath, "====================================================")
    print("=====================================================================================================")
    time.sleep(1)
    with open(fullPath, 'r') as f:
        fileText = []
        for line in f:
            #print(line, end="")
            fileText.append(line)

    bufferSize = 30 # display 60 lines at a time
    lineno = 1
    for x in fileText:
        if bufferSize == 0:
            bufferSize = 30
            # sleep for 10 milliseconds. 
            time.sleep(0.5)
            print("-------------------------------------")
        print(lineno, "  ", x, end="")
        lineno += 1
        bufferSize -= 1
