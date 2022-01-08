# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    node = (problem.getStartState(),[]) #Create node with the state and path
    if problem.isGoalState(node[0]): #if the start state fulfill the goal 
        return []
    frontier = Stack() #LIFO for depth searching
    frontier.push(node) #push node in stack
    explored = set() #Visited states/nodes
    while(True): #infinity loop 
        if frontier.isEmpty(): #If the algorithm has searched every state then return error
            return []
        node = frontier.pop() #Pop the last node in the stack
        explored.add(node[0]) #Add it in visited states
        if problem.isGoalState(node[0]):#If the state does fulfill the goal the return its path
            return node[1]
        successor = problem.getSuccessors(node[0]) #Get every near state and the action to go there
        for su in successor: #for every state in successor
            if su[0] not in explored and su[0] not in frontier.list: #If the state is not visited or in stack
                path = node[1] + [su[1]] #Create new pathz
                frontier.push((su[0],path)) #Push new node in stack
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    node = (problem.getStartState(),[]) #Create node with the state and path
    if problem.isGoalState(node[0]):#if the start state fulfill the goal 
        return []
    frontier = Queue() #FIFO for breadth searching
    frontier.push(node) #push node in Queue
    explored = []  #Visited states/nodes
    while True: #infinity loop 
        if frontier.isEmpty(): #If the algorithm has searched every state then return error                                                 
            return []
        node = frontier.pop() #Pop the first node in the queue
        explored.append(node[0])#Add it in visited states
        if problem.isGoalState(node[0]): #If the state does fulfill the goal the return its path
            return node[1]
        successor = problem.getSuccessors(node[0]) #Get every near state and the action to go there
        for su in successor:
            if su[0] not in (i[0] for i in frontier.list) and su[0] not in explored: #If the state is not visited or in queue
                path = node[1] + [su[1]] #Create new path
                frontier.push((su[0],path))#Push new node in queue
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    node = (problem.getStartState(),[]) #Create node with the state and path
    if problem.isGoalState(node[0]): #if the start state fulfill the goal 
            return []
    frontier = PriorityQueue() #Priority queue data structure
    frontier.push(node,0) #Push node in PQueue
    explored = set() #Visited states/nodes
    while True: #infinity loop 
        if frontier.isEmpty(): #If the algorithm has searched every state then return error
            return []
        node = frontier.pop() #Pop the node with the lowest priority
        if problem.isGoalState(node[0]): #If the state does fulfill the goal the return its path
            return node[1]
        explored.add(node[0]) #Add it in visited states
        successor = problem.getSuccessors(node[0]) #Get every near state and the action to go there
        for su in successor:
            if (su[0] not in (i[2][0] for i in frontier.heap)) and (su[0] not in explored): #If the state is not visited or in queue the create node
                path = node[1] + [su[1]] #Create new path
                pri = problem.getCostOfActions(path) #Calculate the priority of the node
                frontier.push((su[0],path),pri) #Push new node in pqueue
            elif (su[0] in (i[2][0] for i in frontier.heap)) and (su[0] not in explored): #If the state is in Pq and not visited
                path = node[1] + [su[1]] #Create new path
                for y in frontier.heap: #Find previous priority
                    if y[2][0] == su[0]:
                        prepri = problem.getCostOfActions(y[2][1])
                pri = problem.getCostOfActions(path) #Calculate its new priority
                if prepri > pri: #If new priority is lower than previous the update it with the new one
                    frontier.update((su[0],path),pri)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    node = (problem.getStartState(),[]) #Create node with the state and path
    if problem.isGoalState(node[0]): #if the start state fulfill the goal 
            return []
    frontier = PriorityQueue() #Priority queue data structure
    frontier.push(node,0) #Push node in PQueue
    explored = [] #Visited states/nodes
    while True: #infinity loop 
        if frontier.isEmpty(): #If the algorithm has searched every state then return error
            return []
        node = frontier.pop() #Pop the node with the lowest priority
        if problem.isGoalState(node[0]): #If the state does fulfill the goal the return its path
            return node[1]
        explored.append(node[0])#Add it in visited states
        successor = problem.getSuccessors(node[0]) #Get every near state and the action to go there
        for su in successor:
            if (su[0] not in (i[2][0] for i in frontier.heap)) and (su[0] not in explored): #If the state is not visited or in queue the create node
                path = node[1] + [su[1]] #Create new path
                cost = problem.getCostOfActions(path) + heuristic(su[0],problem) #Calculate the priority/cost of the node with the addition of cost by the heuristic function
                frontier.push((su[0],path),cost) #Push new node in queue
            elif (su[0] in (i[2][0] for i in frontier.heap)) and (su[0] not in explored): #If the state is in Pq and not visited
                path = node[1] + [su[1]] #Create new path
                for y in frontier.heap:  #Find previous priority/cost
                    if y[2][0] == su[0]:
                        precost = problem.getCostOfActions(y[2][1]) + heuristic(su[0],problem)
                cost = problem.getCostOfActions(path) + heuristic(su[0],problem) #Calculate its new priority
                if precost > cost: #If new priority.cost is lower than previous the update it with the new one
                    frontier.update((su[0],path),cost)
    util.raiseNotDefined()



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
