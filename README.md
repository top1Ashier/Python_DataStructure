---
ebook:
  theme: github-light.css
  title: Python数据结构
  authors: Ashier
---
[TOC]
# Python 数据结构与算法

## 线性结构(Linear structure)
- 线性结构是一种有序数据项的集合，其中每个数据项都有唯一的前驱和后继。线性结构总有两端，在不同的情况下，两端的称呼不一样。两端的称呼并不是关键，不同线性结构的关键区别在于数据项增减的方式。
- 四种常用的线性结构:*栈(Stack),队列(Queue),双端队列(Deque),列表(List)*

### 栈(Stack)
 栈是一种有次序的数据项的集合，在栈中，数据项的加入和移除都仅发生在一端。距离栈底越近的元素，留在栈中的时间越久，常被称为“后进先出(LIFO)”。

#### 栈的抽象数据类型(ADT of Stack):
 - **Operation**:
    - Stack():栈的初始化，创建一个空栈，不包含任何数据项。
    - push(item):把item加入栈顶，无返回值。
    - pop():将栈顶数据项移除，并返回，栈被修改。
    - peek():返回栈顶的数据项但不移除，栈不被修改。
    - isEmpty():返回栈是否为空栈。
    - size():返回栈中有多少个数据项。

**python3简易实现(List尾端为栈顶)**:
```python
class Stack:
    def __init__(self):
        self.items=[]
    def push(self,item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[-1]
    def isEmpty(self):
        return self.items==[]
    def size(self):
        return len(self.items)
```

#### 栈的应用示例
##### 简单括号匹配
**算法描述**:  
    1. 生成一个空栈
    2. 从右到左依次取括号
    3. 取到左括号将左括号存入栈中，回到**2**取到右括号判断栈是否为空，是的话则匹配失败，程序结束，否则，移除栈顶元素后回到**2**，若所有元素已经取完，判断栈是否为空
    4. 栈空则匹配成功，程序结束

```python
 def parcheck(pStr):
    s=Stack()
    index=0
    judge=True
    while index<len(pStr) and judge:
        if pStr[index]=='(':
            s.push(pStr[index])
        else:
            if s.isEmpty():
                judge=False
            else:
                s.pop()
        index=index+1
    if judge and s.isEmpty():
        return True
    else:
        return False
```

*通用括号匹配*:

```python
def Gparcheck(pStr):
    s=Stack()
    index=0
    judge=True
    while index<len(pStr) and judge:
        if pStr[index] in '{[(':
            s.push(pStr[index])
        else:
            if s.isEmpty():
                judge=False
            else:
                if matchp(s.peek(),pStr[index]):
                    s.pop()
                else:
                    judge=False
        index=index+1
    if judge and s.isEmpty():
        return True
    else:
        return False
def matchp(p1,p2):
    opens=['{','[','(']
    closers=['}',']',')']
    return opens.index(p1)==closers.index(p2)
```

##### 表达式中缀转后缀

**算法描述**:
从左到右逐个字符扫描中缀表达式的过程中，用一个栈来暂存未处理的操作符，当遇到一个新的操作符，和栈顶的操作符比较优先级，再进行处理。

```python
from string import ascii_uppercase,digits
def Infix2Postfix(infixexpr):
    '''优先级标识'''
    prec={'*':3,'/':3,'+':2,'-':2,')':1}
    opStack=Stack()
    postfixList=[]
    tokenList=infixexpr.split()
    for token in tokenList:
        if token in ascii_uppercase or token in digits:
            postfixList.append(token)
        elif token=='(':
            opStack.push(token)
        elif token==')':
            topToken=opStack.pop()
            while topToken!='(':
                postfixList.append(topToken)
                topToken=opStack.pop()
        else:
            while(not opStack.isEmpty()) and \
            prec[opStack.peek()]>=prec[token]:
                postfixList.append(obStack.pop())
            opStack.push(token)
    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
    return " ".join(postfixList)
```
### 队列(Queue)
队列是一种有次序的数据集合，其特征是新数据项的添加总发生在一端尾端(rear),而现存数据项的移除总发生在另一端首端(front)。新加入的数据项必须在数据集末尾等待，等待时间最长的数据项则是队首,该次序安排原则称为先进先出(FIFO)。

#### 队列的抽象数据类型(ADT of Queue):
- **Operation**:
    - Queue():创建一个空队列对象，返回值为Queue对象。
    - enqueue(item):将数据项item添加到队尾，无返回值。
    - dequeue():从队首移除数据项，返回值为队首数据项，队列被修改。
    - isEmpty():测试是否为空队列，返回bool值。
    - size():返回队列中数据项的个数。

 **python3简易实现(List首端为队尾，末端为队首)**

```python
 class Queue:
    def __init__(self):
        self.items=[]
    def enqueue(self,item):
        self.items.insert(0,item)
    def dequeue(self):
        return self.items.pop()
    def isEmpty(self):
        return self.items==[]
    def size(self):
        return len(self.items)
```

#### 队列的应用示例

##### 热土豆问题

**算法描述**
模拟程序采用队列来存放所有参加游戏的人名，按照土豆传递方向从队首排到队尾，队首为土豆持有者。模拟开始，只需要将队首的人出队，随即到队尾入队，算是土豆的一次传递，传递num次后，将队首的人移除，不再入队，如此反复，直到队列中剩余1人。

```python
def hotPotato(namelist,num):
    simqueue=Queue()
    for name in namelist:
        simqueue.enqueue(name)
    while simqueue.size()>1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())
        simqueue.dequeue()
    return simqueue.dequeue()
```

##### 打印任务
- 对问题建模
  - 对象：打印任务(提交时间、打印页数)、打印队列(具有FIFO特性的打印任务队列)、打印机(打印速度、是否忙)
  - 过程：生成和提交打印任务，实施打印
  - 模拟时间：在一个时间单位里，对生成打印任务和实施打印两个过程各处理一次

```python 
import random
class Printer: 
    """
    初始化Printer对象
    pagerate:打印速度(minute)
    currentTask:打印任务
    timeRemaining:任务倒计时
    """
    def __init__(self,ppm):
        self.pagerate=ppm   
        self.currentTask=None   
        self.timeRemaining=0    
    """计时器"""
    def tick(self):
        if self.currentTask!=None:
            self.timeRemaining=self.timeRemaining-1
            if self.timeRemaining<=0:
                self.currentTask=None 
    """打印机工作状态"""
    def isBusy(self):
        if self.currentTask!=None:
            return True
        else:
            return False
    """打印新任务"""
    def startNext(self,newtask):
        self.currentTask=newtask
        self.timeRemaining=newtask.getPages()*60/self.pagerate
    
class Task:
    """
    初始化打印任务
    timestamp:任务生成的时间戳
    pages：打印任务的页数
    """
    def __init__(self,time):
        self.timestamp=time
        self.pages=random.randrange(1,21)
    def getStamp(self):
        return self.timestamp
    def getPages(self):
        return self.pages
    """等待时间"""
    def waitTime(self,currenttime):
        return currenttime-self.timestamp
"""生成任务概率控制"""
def newTask():
    tmp=random.randrange(1,181)
    if tmp==180:
        return True
    else:
        return False
"""
模拟
numSecconds:时间限制
pagesPerMinute:打印机一分钟打印页数
"""
def simulation(numSeconds,pagesPerMinute):
    labprinter=Printer(pagesPerMinute)
    printQueue=Queue()
    waitingtimes=[]
    """时间流逝"""
    for currentSecond in range(numSeconds):
        """生成新任务加入队列中"""
        if newTask():
            task=Task(currentSecond)
            printQueue.enqueue(task)
        """打印机空闲，分派任务"""
        if (not labprinter.isBusy()) and (not printQueue.isEmpty()):
            nexttask=printQueue.dequeue()
            waitingtimes.append(nexttask.waitTime(currentSecond))
            labprinter.startNext(nexttask)
        labprinter.tick()
    averageWait=sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d task remaining."%(averageWait,printQueue.size()))
```

### 双端队列(Deque)
双端队列是一种有次序的数据集，跟队列相似，其两端可以称作首，尾端，但其中数据项既可以从队首加入，亦可以从队尾加入；数据项也可以从两端移除。(可以用deque模拟栈，队列)

#### 双端队列的抽象数据类型(ADT of Deque)
- **Operation**
  - Deque():创建一个空双端队列。
  - addFront(item):将item加入队首。
  - addRear(item):将item加入队尾
  - removeFront():从队首移除数据项，返回值为移除的数据项
  - removeRear():从队尾移除数据项，返回值为移除的数据项
  - isEmpty():返回deque是否为空。
  - size():返回deque中包含数据项的个数。

**python3简易实现(List首端为队尾，末端为队首)**
```python
class Deque:
    def __init__(self):
        self.items=[]
    def addFront(self,item):
        self.items.append(item)
    def addRear(self,item):
        self.items.insert(0,item)
    def removeFront(self):
        return self.items.pop()
    def removeRear(self):
        return self.items.pop(0)
    def isEmpty(self):
        return self.items==[]
    def size(self):
        return len(self.items)
```
#### 双端队列的应用示例
##### 回文词(parlindorme)的判定
**算法描述**：
先将需要判定的词从队尾加入deque，再从两端同时移除字符判定是否相同，直到deque中剩下0个或1个字符。

```python
def palchecker(word):
    chardeque=Deque()
    for ch in word:
        chardeque.addRear(ch)
        judge=True
    while chardeque.size()>1:
        if chardeque.removeFront()!=chardeque.removeRear():
            judge=False
    return judge
```
### 无序表(UnOrderList)
#### 无序表的抽象数据类型(ADT of UnOrderList)
- **Operation**:
  - UnOrderList():创建一个空列表。
  - add(item):添加一个数据项到列表中，假设item原先不存在列表中。
  - remove(item):从列表中移除item，列表被修改，item原先应存在列表在中。
  - search(item):在列表中查找item，返回bool值和数据项在表中位置。
  - isEmpty():返回列表是否为空。
  - size():返回列表包含了多少数据项。
  - append(item):添加一个数据项到表末尾，假设item原先不存在于列表中。
  - index(item):返回数据项在表中的位置。
  - insert(pos,item):将数据项插入到位置pos,假设item原先不存在于列表中，同时原列表具有足够多个数据项，能让item占据位置pos。
  - pop():从列表末尾移除数据项，假设原列表至少有1个数据项。
  - pop(pos):移除位置为pos的数据项，假设原列表存在位置pos。

python采用单链表实现无序表
- 单链表实现:
    - 节点Node,
    - 无序表:无序表必须有对第一个节点的引用信息，head始终指向第一个节点

```python
class Node:
    def __init__(self,initdata):
        self.data=initdata
        self.next=None 
        #self.prev=None
    def getData(self):
        return self.data 
    def getNext(self):
        return self.next
    def setData(self,newdata):
        self.data=newdata
    def setNext(self,newnext):
        self.next=newnext
```

```python
class UnorderedList:
    def __init__(self):
        self.head=None 
    """表头添加O(1)"""
    def add(self,item):
        temp=Node(item)
        temp.setNext(self.head)
        self.head=temp
    """表头开始遍历实现O(n)"""
    def size(self)；
        current=self.head
        count=0
        while count!=None:
            count=count+1
            current=current.getNext()
        return count
    """查找O(n)"""
    def search(item):
        current=self.head
        count=0
        found=False
        while not found:
            if current.getData()==item:
                found=True
            else:
                current=current.getNext()
            count=count+1
        return found,count
    """先search到，再remove(考虑头节点是要remove的值的情况)"""
    def remove(self,item):
        current=self.head
        prev=None 
        found=False
        while not found:
            if current.getData()==item:
                found=True
            else:
                prev=current
                current=current.getNext()
        if prev==None:
            self.head=current.getNext()
        else:
            prev.setNext(current.getNext())
    def isEmpty(self):
        return self.head==None 
    def  v
```
*有序表:类似无序表，区别于多了大小的比较,search,add方法需要进行修改，search可利用有序表有序排列的特性，节省查找的时间。*

## 递归
**递归三定律**:基本结束条件，如何减小规模，调用自身
### 递归调用的实现
当一个函数被调用的时候，系统会把调用时的现场数据亚茹到系统调用栈。每次调用，压入栈的现场数据称为栈帧。当函数返回时，要从调用栈的栈顶取得返回地址，恢复现场，弹出栈帧，按地址返回。
*Python中的递归深度限制,RecursionError,递归的层数太多，系统调用栈的容量有限。要检查基本结束条件，或者是算法导致向基本条件演进太慢。Python中sys模块可以调整最大递归深度sys.setrecursionlimit()。*
### 递归示例
#### 整数转换为任意进制
```python
def toStr(n,base):
    convertString='012456789ABCDE'
    if n<base:
        return convertString[n]
    else:
        return toStr(n//base,base)+convertString[n%base]
```
#### 递归可视化:分形树(Fractal tree)
**分形**:”一个粗糙或零碎的几何形状，可以分成数个部分，且每一部分都(至少近似地)是整体缩小后的形状“，即具有*自相似*的性质。
```python
import turtle
def tree(branch_len):
    if branch_len>5:
        t.forward(branch_len)
        t.right(20)
        tree(branch_len-15)
        t.left(40)
        tree(branch_len-15)
        t.right(20)
        t.backward(branch_len)
t=turtle.Turtle()
t.left(90)
t.penup()
t.backward(100)
t.pendown()
t.pencolor('red')
t.pensize(2)
tree(75)
t.hideturtle()
turtle.done()
```

#### 递归可视化:谢尔宾斯基三角形(Sierpinski triangle)
分型构造，平面称为谢尔宾斯基三角形，立体称为谢尔宾斯基金字塔，真正的谢尔宾斯基三角形是完全不可见的，其面积为零，但周长无穷，是介于一维和二维之间的分数维(约1.585维)构造。

```python
import turtle
def Sierpinski(degree,points):
    colormap=['blue','red','green','white','yellow','orange']
    drawTriangle(points,colormap[degree])
    if degree>0:
        Sierpinski(degree-1,
        {'left':points['left'],
        'top':getMid(points['left'],points['top']),
        'right':getMid(points['left'],points['right'])})
        Sierpinski(degree-1,
        {'left':getMid(points['left'],points['top']),
        'top':points['top'],
        'right':getMid(points['top'],points['right'])})
        Sierpinski(degree-1,
        {'left':getMid(points['left'],points['right']),
        'top':getMid(points['top'],points['right']),
        'right':points['right']})
def drawTriangle(points,color):
    t.fillcolor(color)
    t.penup()
    t.goto(points['top'])
    t.pendown()
    t.begin_fill()
    t.goto(points['left'])
    t.goto(points['right'])
    t.goto(points['top'])
    t.end_fill()
def getMid(p1,p2):
    return((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

t=turtle.Turtle()
points={'left':(-200,-100),
        'top':(0,200),
        'right':(200,-100)}
Sierpinski(5,points)
turtle.done()
```

#### 递归可视化:汉诺塔问题
**算法描述**:将盘片塔从开始柱，经由中间柱，移动到目标柱:首先将上层N-1个盘片的盘片塔，从开始1柱，经由目标柱，移动到中间柱；然后将第N个(最大的)盘片，从开始柱，移动到目标柱；最后将放置在中间柱的N-1个盘片的盘片塔，经由开始柱，移动到目标柱。

```python 
def moveTower(height,fromPole,withPole,toPole):
    if height>=1:
        moveTower(height-1,fromPole,toPole,withPole)
        moveDisk(height,fromPole,toPole)
        moveTower(height-1,withPole,fromPole,toPole)
def moveDisk(disk,fromPole,toPole):
    print(f"Moving disk[{disk}] from {fromPole} to {toPole}")
```
## 查找和排序算法

### 顺序查找(Sequential Search)
无序表中时间复杂度平均情况:O(n)
无序表查找代码:
```python
def sequentialSearch(alist,item):
    pos=0
    found=false
    while pos<len(alist) and not found:
        if alist[pos]==item:
            found=True
        else:
            pos=pos+1
    return found
```
有序表中时间复杂度平均情况O(n):
有序表查找代码:
```python
def orderedSequentialSearch(alist,item):
    pos=0
    found=True
    stop=False
    while pos<len(alist) and not found and not stop:
        if alist[pos]==item:
            found=True
        else:
            if alist[pos]>item:
                stop=True
            else:
                pos=pos+1
    return found
```

### 二分查找(Binary Search)
*适用于有序表*,时间复杂度O($logn$)
非递归实现:
```python
def binarysearch(alist,item):
    first=0
    last=len(alist)-1
    found=False
    while first<=kast and not found:
        midpoint=(first+last)//2
        if alist[midpoint]==item:
            found=True
        else:
            if item<alist[midppoint]:
                last=midpoint-1
            else:
                first=midpoint+1
    return found
```
递归实现:
```python
def binarySearch(alist, item):
    if len(alist) == 0:
        return False
    else:
        midpoint = len(alist)//2
        if alist[midpoint]==item:
          return True
        else:
          if item<alist[midpoint]:
            return binarySearch(alist[:midpoint],item)  #切片时间复杂度为O(K)
          else:
            return binarySearch(alist[midpoint+1:],item)
```

### 冒泡排序(Bubble Sort)
**算法描述**:n-1趟相邻两两比较遍历列表，大小排序不符合要求就交换次序。
时间复杂度O($n^2$)
优点:无需额外存储空间开销，适应性好。
```python
def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                alist[i],alist[i+1]=alist[i+1],alist[i]
```
改进方法:检测每趟比对是否发生过交换，提前确定排序是否完成，完成则结束。(*时间复杂度不变*)
```python
 def shortBubbleSort(alist):
    exchanges = True
    passnum = len(alist)-1
    while passnum > 0 and exchanges:
       exchanges = False
       for i in range(passnum):
           if alist[i]>alist[i+1]:
               exchanges = True
               temp = alist[i]
               alist[i] = alist[i+1]
               alist[i+1] = temp
       passnum = passnum-1
```

### 选择排序(Selection Sort)
**算法描述**:多趟比对，每趟都使最大项或最小项就位。
时间复杂度:O($n^2$)，但相比冒泡排序交换次数减少为O(n)。
```python
def selectionSort(alist):
   for fillslot in range(len(alist)-1,0,-1):
       positionOfMax=0
       for location in range(1,fillslot+1):
           if alist[location]>alist[positionOfMax]:
               positionOfMax = location
       alist[fillslot],alist[positionOfMax] = alist[positionOfMax],alist[fillslot]
```
### 插入排序(Insertion Sort)
**算法描述**:列表中元素依次按大小插入子列表中。
时间复杂度:最坏O($n^2$)，最好O($n$)
```python
def insertionSort(alist):
   for index in range(1,len(alist)):

     currentvalue = alist[index]
     position = index

     while position>0 and alist[position-1]>currentvalue:
         alist[position]=alist[position-1]
         position = position-1

     alist[position]=currentvalue
```

### 归并排序(Merge Sort)
**算法描述**:归并排序是递归算法，是将数据表持续分裂为两半，对两半分别进行归并排序。
时间复杂度:分裂O($logn$),合并O($n$),总O($nlogn$),但有更多的存储空间被使用。
```python
def mergeSort(alist):
    print("Splitting ",alist)
    if len(alist)>1:    #基本结束条件
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]
        #递归调用
        mergeSort(lefthalf)
        mergeSort(righthalf)
        i=0
        j=0
        k=0
        while i<len(lefthalf) and j<len(righthalf):
            if lefthalf[i]<righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i<len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j<len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",alist)
```
更易读,更Pythonic
```python
def merge_sort(lst):
    #基本结束条件
    if len(lst)<=1:
        return lst
    #分解问题并递归调用
    middle=len(lst)//2
    left=merge_sort(lst[:middle])
    right=merge_sort(lst[middle:])
    #合并左右
    merged=[]
    while left and right:
        if left[0]<=right[0]:
            merged.append(left.pop(0))
        else:
            merged.append(right.pop(0))
    merged.extend(right if right else left)
    return merged
```

### 快速排序(Qucik Sort)
**算法描述**:(递归实现)找到标准(最好是中位数)分成两半，一半大于该标准，一半小于该标准。
时间复杂度:正常O(n$logn$)，极端O($n^2$)关键在于中值的选取。
```python
def quickSort(alist):
   quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
   if first<last:

       splitpoint = partition(alist,first,last)

       quickSortHelper(alist,first,splitpoint-1)
       quickSortHelper(alist,splitpoint+1,last)

def partition(alist,first,last):
   pivotvalue = alist[first]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and \
               alist[leftmark] <= pivotvalue:
           leftmark = leftmark + 1

       while alist[rightmark] >= pivotvalue and \
               rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark]
           alist[leftmark] = alist[rightmark]
           alist[rightmark] = temp
```

### 哈希表(Hash table)
哈希表是一种数据集，其中数据项的存储方式尤其有利于*快速查找定位*，哈希表每一个存储位置，称为*槽(slot)*，可以用来保存数据项，每个槽有一个唯一的名称。实现从数据项到存储槽名称转换的函数，称为*哈希函数(hash function)。*如果一个哈希函数能把每个数据项映射到不同的槽里，称为完美哈希函数。负载因子(load factor):负载因子 = 总键值对数 / 槽个数。
**优化思路**:
- 扩大哈希表的容量,但对数据项过大的情况不适用
- 哈希函数具有以下性质:冲突最小(近似完美)，计算难度低(额外开销小)，充分分散数据项(节约空间)


**作为"指纹"函数所需要具有的特性**:压缩性(任意长度的数据，得到"指纹"的长度是固定的)、易计算性(计算容易)、抗修改性(对原数据微小的变动，都会引起"指纹"大改变)、抗冲突性(找到不同数据相同"指纹"很困难)。

**哈希函数冲突解决方案**:
- 开放定址(open adderssing):为冲突的数据项再找一个开放的空槽来保存。
    - 线性探测(linear probing):向后逐个槽寻找。缺点:有聚集的趋势，影响其他值存入。
    - 跳跃式探测(skip probing):注意skip的值不能被哈希表的大小整除，可以吧1哈希表大小设置为素数。
    - 二次探测(quadratic probing):不再固定skip的值，而是逐步增加skip的值。
- 数据项链(Chaining):将容纳单个数据项的槽扩展为容纳数据项集合。缺点:随着冲突项的增加，查找时间也会增加。


python哈希函数库 *hashlib*

#### 哈希函数设计
- 折叠法
基本步骤:将数据项按照位数分为若干段，再将几段数字相加，最后对哈希表大小求余，得到哈希值。
- 平方取中法
基本步骤:首先将数据项做平方运算，然后取平方数中间两位，再对哈希表大小求余。
- 非数项
基本步骤:转ASCII码，累加，对散列表大小求余。
    ```python
    def hash(astring,tablesize):
        sum=0
        for pos in range(len(astring)):
            sum=sum+ord(astring[pos])
        return sum%tablesize
    ```
### 映射抽象数据类型及其Python实现
z **Operation**:
    - Map():创建一个空映射，返回空映射对象。
    - put(key,val):将key-val关联对加入映射中，如果key已存在，将val替换旧关联值。
    - get(key):给定key，返回关联的数据值，如不存在，则返回None。
    - del:通过de map[key]的语句形式删除key-val关联。
    - len()返回映射中key-val关联的数目。
    - in:通过key in map的语句形式，返回key是否存在于关联中，返回bool值
```python 
class HashTable:
    def __init__(self):
        self.size=11
        self.slots=[None]*self.size
        self.data=[None]*self.size
    def hashfunc(self,key):
        return key%self.size
    def rehash(self,oldhash):
        return (oldhash+1)%self.size
    def put(self,key,data):
        hashvalue=self.hashfunc(key)
        if self.slots[hashvalue]==None:
            self.slots[hashvalue]=key 
            self.data[hashvalue]=data 
        else:
            if self.slots[hashvalue]==key:
                self.data[hashvalue]=data 
            else:
                nextslot=self.rehash(hashvalue)
                while self.slots[nextslot]!=None and \
                        self.slots[nextslot]!=key:
                    nextslot=self.rehash(nextslot)
                if self.slots[nextslot]==None:
                    self.slots[nextslot]=key 
                    self.data[nextslot]=data 
                else:
                    self.data[nextslot]=data
    def get(self,key):
        startslot=self.hashfunc(key)
        data=None 
        stop=False
        found=False
        position=startslot
        while self.slots[position]!=None and not found and not stop:
            if self.slots[position]==key:
                found=True
                data=self.data[position]
            else:
                position=self.rehash(position)
                if position==startslot:
                    stop=True
        return data
    def __getitem__(self,key):
        return self.get(key)
    def __setitem__(self,key,data):
        self.put(key,data)
```

## 非线性结构
### 树
**树结构相关术语**
- *节点Node*:组成树的基本部分。
- *边Edge*:边是组成树的另一个基本部分。
- *根root*:树中唯一一个没有入边的节点。
- *路径PATH*:由边连接在一起的节点的有序列表。
- *子节点Children*:入边均来自于同一个节点的若干节点，称为这个节点的子节点。
- *父节点Parent*:一个节点是其所有出边所连接节点的父节点。
- *兄弟节点Sibling*:具有同一个父节点的节点之间称为兄弟节点。
- *子树Subtree*:一个节点和其所有子孙节点，以及相关边的集合。
- *叶节点Leaf*:没有子节点的节点称为叶节点。
- *层级Level*:从根节点开始到达一个节点的路径，所包含边的数量。
- *高度height*:树中所有节点的最大层级称为树的高度。

#### 树的实现
##### 嵌套列表法
```python
def BinaryTree(r):
    return [r,[],[]]

def insertL(root,newBranch):
    t=root.pop(1)
    if len(t)>1:
        root.insert(1,[newBranch,t,[]])
    else:
        root.insert(1,[newBranch,[],[]])
    return root

def insertR(root,newBranch):
    t=root.pop(2)
    if len(t)>1:
        root.insert(2,[newBranch,[],t])
    else:
        root.insert(2,[newBranch,[],[]])
    return root

def getRootval(root):
    return root[0]

def setRootval(root,newVal):
    root[0]=newVal

def getLeftChild(root):
    return root[1]

def getRightChild(root):
    return root[2]
```

##### 节点链接法
每个节点除了保存根节点的数据项，以及指向左右子树的链接。
```python
class BinaryTree:

    def __init__(self,rootObj):
        self.key=rootObj
        self.leftChild=None 
        self.rightChild=None 
    
    def insertL(self,newNode):
        if self.leftChild==None:
            self.leftChild=BinaryTree(newNode)
        else:
            temp=BinaryTree(newNode)
            temp.leftChild=self.leftChild
            self.leftChild=temp
    
    def insertR(self,newNode):
        if self.rightChild==None:
            self.rightChild=BinaryTree(newNode)
        else:
            temp=BinaryTree(newNode)
            temp.rightChild=self.rightChild
            self.rightChild=temp
            
    def getLeftChild(self):
        return self.leftChild

    def getRightChild(self):
        return self.rightChild
```

#### 树的应用:表达式解析
譬如自底向上，运算优先级递减。
目标:从全括号表达式构建表达式解析树，利用表达式解析树对表达式求值，从表达式解析树恢复原表达式的字符串形式。
构建过程:
- 创建空树，当前节点为根节点
- 读入'(',创建左子节点
- 读入操作数，当前节点设置为操作数，上升到父节点
- 读入操作符,当前节点设置该操作符，创建右子节点，当前节点下降
- 读入')'，上升到父节点
```python
class Stack:
    def __init__(self):
        self.items=[]
    def push(self,item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[-1]
    def isEmpty(self):
        return self.items==[]
    def size(self):
        return len(self.items)

class BinaryTree:
    def __init__(self,rootObj):
        self.key=rootObj
        self.leftChild=None 
        self.rightChild=None 
    def insertL(self,newNode):
        if self.leftChild==None:
            self.leftChild=BinaryTree(newNode)
        else:
            temp=BinaryTree(newNode)
            temp.leftChild=self.leftChild
            self.leftChild=temp
    def insertR(self,newNode):
        if self.rightChild==None:
            self.rightChild=BinaryTree(newNode)
        else:
            temp=BinaryTree(newNode)
            temp.rightChild=self.rightChild
            self.rightChild=temp
    def getLeftChild(self):
        return self.leftChild
    def getRightChild(self):
        return self.rightChild
    def setRootVal(self,newVal):
        self.key=newVal
    def getRootVal(self):
        return self.key
        
def buildParseTree(fpexp):
    fplist=fpexp.split()
    pStack=Stack()
    eTree=BinaryTree('')
    pStack.push(eTree)
    currentTree=eTree
    for i in fplist:
        if i=='(':
            currentTree.insertL('')
            pStack.push(currentTree)
            currentTree=currentTree.getLeftChild()
        elif i not in ['+','-','*','/']:
            currentTree.setRootVal(int(i))
            parent=pStack.pop()
            currentTree=parent
        elif i in ['+','-','*','/']:
            currentTree.setRootVal(i)
            currentTree.insertR('')
            pStack.push(currentTree)
            currentTree=currentTree.getRightChild()
        elif i==')':
            currentTree=pStack.pop()
        else:
            raise ValueError
    return eTree

#计算一定要加括号譬如'( ( 3 + 5 ) * ( 7 - 4 ) )' 而不是'( 3 + 5 ) * ( 7 - 4 )'
#本质是后序遍历
import operator
def evaluate(parseTree):
    opers={'+':operator.add,'-':operator.sub, \
            '*':operator.mul,'/':operator.truediv}
    leftChild=parseTree.getLeftChild()
    rightChild=parseTree.getRightChild()

    if leftChild and rightChild:
        fn=opers[parseTree.getRootVal()]
        return fn(evaluate(leftChild),evaluate(rightChild))
    else:
        return parseTree.getRootVal()

```

#### 树的遍历(Tree Traversals)
##### 前序遍历(preorder)
先访问根节点，再递归地前序访问，最后前序访问右子树。
```python
def preOrder(tree):
    if tree:
        print(tree.getRootVal())
        preOrder(tree.getLeftChild())
        preOrder(tree.getRightChild())
```

##### 中序遍历(inorder)
先递归的中序访问左子树，再访问根节点，最后中序访问右子树。
```python
def inOrder(tree):
    if tree:
        inOrder(tree.getLeftChild())
        print(tree.getRootVal())
        inOrder(tree.getRightChild())
```


##### 后序遍历(postorder)
先递归地访问左子树，再后序访问右子树，最后访问根节点。
```python
def postOrder(tree):
    if tree:
        postOrder(tree.getLeftChild())
        postOrder(tree.getRightChild())
        print(tree.getRootVal())
```

#### 二叉堆(Binary Heap)实现优先队列(Priority Queue)
二叉堆逻辑结构类似二叉树(使用完全二叉树)，却是用非嵌套的列表实现的。二叉堆可以使优先队列的入队，出队时间复杂度都保持在$O(nlogn)$。
二叉堆的抽象数据类型(ADT of Binary Heap)(*最小堆*)
- **Operation**
    - BinaryHeap():创建一个空的二叉堆。
    - insert(k):将新key加入堆中。
    - findMin():返回堆中的最小项，最小项仍保留在堆中。
    - delMin():返回堆中的最小项，并从堆中删除。
    - isEmpty():返回堆是否为空。
    - size():返回堆中key的个数。
    - bulidHeap(list):从一个key列表(无序表)创建新堆。

完全二叉树:叶节点最多只出现在最底层和次底层，而且最底层的叶节点都连续集中在最左边，每个内部节点都有两个子节点，最多可有一个节点例外。
堆次序(Heap Order):任何一个节点x，其父节点p中的key均小于x中的key。
```python
class BinaryHeap:
    def __init__(self):
        self.heaplist=[0]   #为后面可以用简单的乘除法
        self.currentSize=0

    #key上浮
    def percUp(self,i):
        while i//2>0:
            if self.heaplist[i]<self.heaplist[i//2]:
                self.heaplist[i//2],self.heaplist[i]= \
                    self.heaplist[i],self.heaplist[i//2]
            i=i//2

    #key下沉
    def percDown(self,i):
        while (i*2)<=self.currentSize:
            mc=self.minChild(i)
            if self.heaplist[i]>self.heaplist[mc]:
                self.heaplist[i],self.heaplist[mc]= \
                    self.heaplist[mc],self.heaplist[i]
            i=mc
    
    def minChild(self,i):
        if i*2+1>self.currentSize:
            return i*2
        else:
            if self.heaplist[i*2]<self.heaplist[i*2+1]:
                return i*2
            else:
                return i*2+1
            self.heaplist[i],self.heaplist[mc]= \
                    self.heaplist[mc],self.heaplist[i]
            i=mc
    
    def minChild(self,i):
        if i*2+1>self.currentSize:
            return i*2
        else:
            if self.heaplist[i*2]<self.heaplist[i*2+1]:
                return i*2
            else:
                return i*2+1

    def insert(self,k):
        self.heaplist.append(k)
        self.currentSize=self.currentSize+1
        self.percUp(self.currentSize)

    def delMin(self):
        retVal=self.heaplist[1]
        self.heaplist[1]=self.heaplist[self.currentSize]
        self.currentSize=self.currentSize-1
        self.heaplist.pop()
        self.percDown(1)
        return retVal

    #用insert()时间复杂度会是O(nlogn)
    #故用下沉法precDown()时间复杂度为O(n)
    def buildHeap(self,alist):
        i=len(alist)//2
        self.currentSize=len(alist)
        self.heaplist=[0]+alist[:]
        print(len(self.heaplist),i)
        while i>0:
            print(self.heaplist,i)
            self.percDown(i)
            i=i-1
        print(self.heaplist,i)

    def isEmpty(self):
        return self.currentSize==0
    
    def size(self):
        return self.currentSize
```
#### 二叉查找树(Binary Search Tree)
**性质:比父节点小的key都出现在左子树，比节点大的key都出现在右子树。**

```python
class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key=key
        self.payload=val
        self.leftChild=left
        self.rightChild=right
        self.parent=parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and \
            self.parent.leftChild==self

    def isRightChild(self):
        return self.parent and \
            self.parent.rightChild==self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key=key
        self.payload=value
        self.leftChild=lc
        self.rightChild=rc
        if self.hasLeftChild():
            self.leftChild.parent=self
        if self.hasRightChild():
            self.rightChild.parent=self

     def __iter__(self):
        if self:
            if self.hasLeftChild():
                for elem in self.leftChild:
                    yield elem
            yield self.key
            if self.hasRightChild():
                for elem in self.rightChild:
                    yield elem
    def insert(self,k):
        self.heaplist.append(k)
        self.currentSize=self.currentSize+1
        self.percUp(self.currentSize)

    def delMin(self):
        retVal=self.heaplist[1]
        self.heaplist[1]=self.heaplist[self.currentSize]
        self.currentSize=self.currentSize-1
        self.heaplist.pop()
        self.percDown(1)
        return retVal

    #用insert()时间复杂度会是O(nlogn)
    #故用下沉法precDown()时间复杂度为O(n)
    def buildHeap(self,alist):
        i=len(alist)//2
        self.currentSize=len(alist)
        self.heaplist=[0]+alist[:]
        print(len(self.heaplist),i)
        while i>0:
            print(self.heaplist,i)
            self.percDown(i)
            i=i-1
        print(self.heaplist,i)

    def isEmpty(self):
        return self.currentSize==0
    
    def size(self):
        return self.currentSize
```
#### 二叉查找树(Binary Search Tree)
**性质:比父节点小的key都出现在左子树，比节点大的key都出现在右子树。**

```python
class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key=key
        self.payload=val
        self.leftChild=left
        self.rightChild=right
        self.parent=parent

    def hasLeftChild(self):
        return self.leftChild

    def hasRightChild(self):
        return self.rightChild

    def isLeftChild(self):
        return self.parent and \
            self.parent.leftChild==self

    def isRightChild(self):
        return self.parent and \
            self.parent.rightChild==self

    def isRoot(self):
        return not self.parent

    def isLeaf(self):
        return not (self.rightChild or self.leftChild)

    def hasAnyChildren(self):
        return self.rightChild and self.leftChild

    def replaceNodeData(self,key,value,lc,rc):
        self.key=key
        self.payload=value
        self.leftChild=lc
        self.rightChild=rc
        if self.hasLeftChild():
            self.leftChild.parent=self
        if self.hasRightChild():
            self.rightChild.parent=self

     def __iter__(self):
        if self:
            if self.hasLeftChild():
                for elem in self.leftChild:
                    yield elem
            yield self.key
            if self.hasRightChild():
                for elem in self.rightChild:
                    yield elem

class BinarySearchTree:
    def __init__(self):
        self.root=None 
        self.size=0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__
    
    def put(self,key,val):
        if self.root:
            self._put(key,val,self.root)
        else:
            self.root=TreeNode(key,val)
        self.size=self.size+1
    
    def _put(key,val,currentNode):
        if key<currentNode.key:
            if currentNode.hasLeftChild():
                self.put(key,val,currentNode.leftChild)
            else:
                currentNode.leftChild=TreeNode(key,val,parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key,val,currentNode.rightChild)
            else:
                currentNode.rightChild=TreeNode(key,val,parent=currentNode)

    def __setitem__(self,k,v):
        self.put(k,v)
    
    def get(self,key):
        if self.root:
            res=self._get(key,self.root)
            if res:
                return res.payload
            else:
                return None 
        else:
            return None

    def _get(self,key,currentNode):
        if not currentNode:
            return None
        elif currentNode.key==key:
            return currentNode
        elif key<currentNode.key:
            return self._get(key,currentNode.leftChild)
        else:
            return self._get(key,currentNode.rightChild)

    def __getitem__(self,key):
        return self.get(key)

    def __contains__(self,key):
        if self._get(key,self.root):
            return True
        else:
            return False
    
    def delete(self,key):
        if self.size>1:
            nodeToRemove=self._get(key,self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size=self.size-1
            else:
                raise KeyError('Error,key not in tree')
        elif self.size==1 and self.root.key==key:
            self.root=None 
            self.size=self.size-1
        else:
            raise KeyError('Error,key not in tree')
        
    def __delitem__(self,key):
        self.delete(key)

    def spliceOut(self):
       if self.isLeaf():
           if self.isLeftChild():
                  self.parent.leftChild = None
           else:
                  self.parent.rightChild = None
       elif self.hasAnyChildren():
           if self.hasLeftChild():
                  if self.isLeftChild():
                     self.parent.leftChild = self.leftChild
                  else:
                     self.parent.rightChild = self.leftChild
                  self.leftChild.parent = self.parent
           else:
                  if self.isLeftChild():
                     self.parent.leftChild = self.rightChild
                  else:
                     self.parent.rightChild = self.rightChild
                  self.rightChild.parent = self.parent

    def findSuccessor(self):
      succ = None
      if self.hasRightChild():
          succ = self.rightChild.findMin()
      else:
          if self.parent:
                 if self.isLeftChild():
                     succ = self.parent
                 else:
                     self.parent.rightChild = None
                     succ = self.parent.findSuccessor()
                     self.parent.rightChild = self
      return succ

    def findMin(self):
      current = self
      while current.hasLeftChild():
          current = current.leftChild
      return current

    #三种情况:1.这个将节点没有子节点;2.这个节点有一个子节点3.这个节点有两个子节点
    def remove(self,currentNode):
         if currentNode.isLeaf(): #leaf
           if currentNode == currentNode.parent.leftChild:
               currentNode.parent.leftChild = None
           else:
               currentNode.parent.rightChild = None
         elif currentNode.hasBothChildren(): #interior
           succ = currentNode.findSuccessor()
           succ.spliceOut()
           currentNode.key = succ.key
           currentNode.payload = succ.payload

         else: # this node has one child
           if currentNode.hasLeftChild():
             if currentNode.isLeftChild():
                 currentNode.leftChild.parent = currentNode.parent
                 currentNode.parent.leftChild = currentNode.leftChild
             elif currentNode.isRightChild():
                 currentNode.leftChild.parent = currentNode.parent
                 currentNode.parent.rightChild = currentNode.leftChild
             else:
                 currentNode.replaceNodeData(currentNode.leftChild.key,
                                    currentNode.leftChild.payload,
                                    currentNode.leftChild.leftChild,
                                    currentNode.leftChild.rightChild)
           else:
             if currentNode.isLeftChild():
                 currentNode.rightChild.parent = currentNode.parent
                 currentNode.parent.leftChild = currentNode.rightChild
             elif currentNode.isRightChild():
                 currentNode.rightChild.parent = currentNode.parent
                 currentNode.parent.rightChild = currentNode.rightChild
             else:
                 currentNode.replaceNodeData(currentNode.rightChild.key,
                                    currentNode.rightChild.payload,
                                    currentNode.rightChild.leftChild,
                                    currentNode.rightChild.rightChild)

```

##### AVL树
AVL的实现中，需要对每个节点跟踪平衡因子(balance factor)参数。平衡因子是根据节点的左右子树的高度来定义的(左右子树高度差$balancefactor=height(leftSubTree)-height(rightSubTree)$)。大于0，左重；小于0，右重；等于0，平衡。

```python
def _put(key,val,currentNode):
        if key<currentNode.key:
            if currentNode.hasLeftChild():
                self.put(key,val,currentNode.leftChild)
            else:
                currentNode.leftChild=TreeNode(key,val,parent=currentNode)
                self.updateBalance(currentNode.leftChild
            if currentNode.hasRightChild():
                self._put(key,val,currentNode.rightChild)
            else:
                currentNode.rightChild=TreeNode(key,val,parent=currentNode)
                self.updateBalance(currentNode.rightChild)
def updateBalance(self,node):
    if node.balanceFactor>1 or node.balanceFactor<-1:
        self.rebalance(node)
        return
    if node.parent!=None:
        if node.isLeftChild():
            node.parent.balanceFactor+=1
        elif node.isRightChild():
            node.parent.balanceFactor-=1
        if node.parent.balanceFactor!=0:
            self.updateBalance(node.parent)

#将不平衡的子树进行旋转rotation(同时更新相关父节点引用，更新旋转后被影响节点的平衡因子)

#左旋
def rotateLeft(self,rotRoot):
    newRoot=rotRoot.rightChild
    rotRoot.rightChild=newRoot.leftChild
    if newRoot.leftChild!=None:
        newRoot.leftChild.parent=rotRoot
    newRoot.parent=rotRoot.parent
    if rotRoot.isRoot():
        self.root=newRoot
    else:
        if rotRoot.isLeftChild():
            rotRoot.parent.leftChild=newRoot
        else:
            rotRoot.parent.rightChild=newRoot
    newRoot.leftChild=rotRoot
    rotRoot.parent=newRoot
    #调整平衡因子
    rotRoot.balanceFactor=rotRoot.balanceFactor+1-min(newRoot.balanceFactor,0)
    newRoot.balanceFactor=newRoot.balanceFactor+1+max(rotRoot.balanceFactor,0)

def reblance(self,node):
    if node.balanceFactor<0:
        if node.rightChild.balanceFactor>0:
            #右子节点左重先右旋右子节点再左旋
            self.rotateRight(node,rightChild)
            self.rotateLeft(node)
        else:
            #直接左旋
            self.rotateLeft(node)
    elif node.balanceFactor>0:
        if node,leftChild.balanceFactor<0:
            #左子节点右重先左旋左子节点再右旋
            self.rotateLeft(node.leftChild)
            self.rotateRight(node.node)
        else:
            #直接右旋
            self.rotateRight(node)
```

AVL树put和get时间复杂度始终为$O(logn)$。

### 图(Graph)
图是比树更为一般的结构，也是由节点和边组成。

- **术语表**:
    - 顶点Vertex(也称节点Node)
    - 边Edge(也称弧Arc)
    - 权重Weight
    - 路径Path
    - 圈Cycle(首尾顶点相同的路径)
一个图可以定义为G=(V,E)还可以添加权重分量。

#### 图的抽象数据类型(ADT of Graph)
- **Operation**:
    - Graph():创建一个空的图。
    - addVertex(vert):将顶点vert加入图中。
    - addEdge(fromVert,toVert):添加有向边。
    - addEdge(fromVert,toVert,weight):添加带权的有向边。
    - getVertex(vKey):查找名称为vKey的顶点。
    - getVertices():返回图中所有顶点列表。
    - in:如其原本意义，返回True或False

图的几种表示方法:
- 邻接矩阵(Adjacency Matrix):简单，但如果图中边数很少效率低下
- 邻接列表(Adjacency List):紧凑高效

```python
class Vertex:
    def __init__(self,key):
        self.id=key
        self.connectedTo={}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr]=weight

    def __str__(self):
        return str(self.id)+' connectedTo: '+\
            str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id
    
    def getWeight(self,nbr):
        return self.connectedTo[nbr]

class Graph:
    def __init__(self):
        self.vertList={}
        self.numVertices=0
    
    def addVertex(self,key):
        self.numVertices=self.numVertices+1
        newVertex=Vertex(key)
        self.vertList[key]=newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None 
        
    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,fromVert,toVert,weight=0):
        if fromVert not in self.vertList:
            nv=self.addVertex(fromVert)
        if toVert not in self.vertList:
            nv=self.addVertex(toVert)
        self.vertList[fromVert].addNeighbor(self.vertList[t],weight)

    def getVertex(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values)
```

#### BFS

#### DFS