# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        listfood = []
        listghost = []

        for foodpos in newFood.asList(): #Create list with the distances from every food
            listfood.append(manhattanDistance(foodpos,newPos))
        if(listfood == []): #If it succeeds our goal then return inf 
            return 10000
        else:
            minfood = min(listfood) #Distance from the closest food
            maxfood = max(listfood) #Distance from the furtherst food
        
        for ghostpos in successorGameState.getGhostStates():#Create list with the distances from every ghost
            listghost.append(manhattanDistance(newPos,ghostpos.getPosition()))
        if listghost!=[]:#if there are ghosts take the minimum distance 
            minghost = min(listghost)
        #The move gets worse as the return value goes up. Also, for pacman its better to get away from ghost and closer to the food
        #So we add the ghost that is the closest to pacman and subtract the distances from the furtherst and closest food. 
        if len(listfood) == 1: #If there is only one food left then minfood = maxfood so it calculates only once
            num = minghost - minfood 
        else:
            num = minghost - maxfood - minfood
        return successorGameState.getScore() + num #Return score with the number we calculated before.


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """ 

    def minvalue(self, gameState,depth,agent):
        if depth == 0 or gameState.isLose() or gameState.isWin(): #If goal state or depth == 0
            return self.evaluationFunction(gameState)
        value = 10000 #Value is infinite
        for a in gameState.getLegalActions(agent): #For every move that ghost can make
            if(agent!=gameState.getNumAgents()-1): #If its not the last ghost then call minvalue again to find the minimum value of all the ghosts
                value = min(value,self.minvalue(gameState.generateSuccessor(agent,a),depth,agent+1)) #Calling minvalue again with the next agent
            else:
                value = min(value,self.maxvalue(gameState.generateSuccessor(agent,a),depth - 1))#If its the last ghost then call maxvalue for next depth
        return value

    def maxvalue(self, gameState,depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():#If goal state or depth == 0
            return self.evaluationFunction(gameState)
        value = -10000 #Value is -infinite
        for a in gameState.getLegalActions(0):#For every move that pacman can make
            value = max(value,self.minvalue(gameState.generateSuccessor(0,a),depth,1))#Find the maximum value and call minvalue that starts with the first ghost.
        return value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        scores = []
        for a in gameState.getLegalActions(0): #For every action that pacman can make start min node and store it in a list
            scores.append(self.minvalue(gameState.generateSuccessor(0,a),self.depth,1)) 
        bestScore = max(scores) #Best score
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] #Find the action with the best score
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return gameState.getLegalActions()[chosenIndex]
    
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def minvalue(self, gameState,depth,agent,alpha,beta):
        if depth == 0 or gameState.isLose() or gameState.isWin(): #If goal state or depth == 0
            return self.evaluationFunction(gameState)
        value = 10000 #Value is infinite
        for a in gameState.getLegalActions(agent): #For every move that ghost can make
            if(agent!=gameState.getNumAgents()-1): #If its not the last ghost then call minvalue again to find the minimum value of all the ghosts
                value = min(value,self.minvalue(gameState.generateSuccessor(agent,a),depth,agent+1,alpha,beta))#Calling minvalue again with the next agent
            else:
                value = min(value,self.maxvalue(gameState.generateSuccessor(agent,a),depth - 1,alpha,beta)) #If its the last ghost then call maxvalue for next depth
            if value < alpha: #If the value is less than the best case for max then return value(pruning)
                return value
            beta = min(beta,value)#Beta is the best case for the ghosts so its the minimum value that it can take
        return value

    def maxvalue(self, gameState,depth,alpha,beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():#If goal state or depth == 0
            return self.evaluationFunction(gameState)
        value = -10000 #Value is -infinite
        for a in gameState.getLegalActions(0):#For every move that pacman can make
            value = max(value,self.minvalue(gameState.generateSuccessor(0,a),depth,1,alpha,beta)) #Find the maximum value and call minvalue that starts with the first ghost.
            if value > beta:#If the value is more than the best case for min then return value(pruning)
                return value
            alpha = max(alpha,value)#Alpha is the best case for pacman so its the maximum value that it can take
        return value


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction  
        """
        "*** YOUR CODE HERE ***"
        scores = []
        a = -1000 #-Infinite
        b = 1000 #Infinite
        for action in gameState.getLegalActions(0):#For every action that pacman can make start min node and store it in a list
            scores.append(self.minvalue(gameState.generateSuccessor(0,action),self.depth,1,a,b)) #Call minvalue to find the nodes-child of pacman node(max)
            bestScore = max(scores)#Max score
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] #Find action that has the bestaction
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            a = max(a,bestScore) #Re-calculate alpha so it can do pruning algorithm for the root (max) 
        return gameState.getLegalActions()[chosenIndex]




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expvalue(self, gameState,depth,agent):
        if depth == 0 or gameState.isLose() or gameState.isWin():#If goal state or depth == 0
            return self.evaluationFunction(gameState)
        value = 0
        for a in gameState.getLegalActions(agent): #For every move that ghost can make
            if(agent!=gameState.getNumAgents()-1): #If its not the last ghost then call expvalue again to find the expected value of all the ghosts
                value = value + self.expvalue(gameState.generateSuccessor(agent,a),depth,agent+1)
            else:
                value = value + self.maxvalue(gameState.generateSuccessor(agent,a),depth - 1)#If its the last ghost then call maxvalue for next depth
        return value

    def maxvalue(self, gameState,depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():#If goal state or depth == 0
            return self.evaluationFunction(gameState)
        value = -100000 #Value is -infinite
        for a in gameState.getLegalActions(0):#For every move that pacman can make
            value = max(value,self.expvalue(gameState.generateSuccessor(0,a),depth,1))#Find the maximum value and call expvalue that starts with the first ghost.
        return value

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        scores = []
        for a in gameState.getLegalActions(0): #For every actiona that pacman can make start expexted node and store it in a list
            scores.append(self.expvalue(gameState.generateSuccessor(0,a),self.depth,1))
        bestScore = max(scores)#Best score
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore] #Find the action with the best score
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return gameState.getLegalActions()[chosenIndex]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newFood = currentGameState.getFood()
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    listfood = []
    listghost = []
    for foodpos in newFood.asList(): #Create list with the distances from every food
        listfood.append(manhattanDistance(newPos,foodpos))
    if (newFood.asList() == []): #If it succeeds our goal then return score
        return currentGameState.getScore()
    else:
        minfood = min(listfood) #Distance from the closest food
        maxfood = max(listfood) #Distance from the furtherst food
    for ghostpos in currentGameState.getGhostStates():#Create list with the distances from every ghost
        listghost.append(manhattanDistance(newPos,ghostpos.getPosition()))
    if (listghost != []):#if there are ghosts take the minimum distance 
        minghost = min(listghost)
    #If pacman had a power-up and has more than 16 moves then try to eat the ghost
    if (sum(newScaredTimes) > 16):
        num = minghost
    #The move gets worse as the return value goes up. Also, for pacman its better to get away from ghost and closer to the food
    #So we add the ghost that is the closest to pacman and subtract the distances from the furtherst and closest food. 
    elif (len(newFood.asList()) == 1):
        num = minghost - minfood   
    else:
        num = minghost - maxfood - minfood
    return currentGameState.getScore() + num#Return score with the number we calculated before.



# Abbreviation
better = betterEvaluationFunction
