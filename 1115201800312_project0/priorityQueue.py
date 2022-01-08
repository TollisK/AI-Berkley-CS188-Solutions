import heapq

class PriorityQueue: 
    def __init__(self): #initialize variables
        self.pq = [] #heap
        self.count = 0 #counter of items in heap
    def push(self,item,priority): #push function
        heapq.heappush(self.pq,(priority,item)) #Creats a tuple for the 2 data and pushes them in the heap
        self.count += 1 
    def pop(self):#pop function
        if(self.isEmpty()==False):#If its not empty
            self.count -= 1
            return heapq.heappop(self.pq)[1] #pops only the item 
        print("error")
        return None
    def isEmpty(self):#If the heap is empty return true
        if(self.count == 0):
            return True
        return False
    def update(self,item,priority): 
        cou = 0
        if(len([aritem for aritem in self.pq if aritem[1] == item]) == 0): #If there is not the item in the list 
            self.push(item,priority) #push the new item with its priority
        
        for pri,it in self.pq:
            if(it == item and pri > priority): #if the item is on the list and has priority higher than the argument then change it
                self.pq[cou] = (priority,item) #change its priority
            cou += 1

def PQSort(array):
    heap = PriorityQueue()
    for item in array: #for every item in the array push it in heap
        heap.push(item,item)
    return [heap.pop() for i in range(len(heap.pq))] #Pop all items from heap

