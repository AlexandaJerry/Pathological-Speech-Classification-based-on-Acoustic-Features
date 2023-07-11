import os
import sys
import subprocess

curPath = os.path.abspath(os.path.dirname(__file__))
rootpath=str(curPath)
## print(rootpath)
syspath=sys.path
depth = rootpath.replace('\\', '/').count("/") - 1
## print(depth)
sys.path=[]
sys.path.append(rootpath)#将工程根目录加入到python搜索路径中
# sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[depth]!="."])
# #将工程目录下的一级目录添加到python搜索路径中
# sys.path.extend(syspath)

a1 = "python main.py"

def main():
    p1 = os.system(a1)
    
if __name__ == '__main__':
    main()