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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()
        
        ghosts = []
        foods = []

        radiusSum = 0

        # Get the positions of the food, calc distance, then append them to a list
        for food in newFood.asList():
            foods.append(util.manhattanDistance(newPos, food))
        # Get the positions of ghosts, calc distance, then put it in a list
        for ghost in ghostPositions:
            nextGhost = util.manhattanDistance(newPos, ghost)
            ghosts.append(nextGhost)
        # No foods left? You're gucci
        if len(foods) == 0:
            return 9999999

        # If the nearest ghost isn't that close, don't worry about life
        if (min(ghosts) > 3):
            radiusSum = 10

        foodRecip = 1/float(min(foods)) #idk why this would help but spec says soooo
        minGhost = min(ghosts)

        #No ghosts? Just move along. 
        if len(ghosts) == 0: 
           return successorGameState.getScore() + foodRecip
        scaredTimes = sum(newScaredTimes)

        return radiusSum + successorGameState.getScore() + scaredTimes  + (foodRecip * minGhost)

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

        totalAgents = gameState.getNumAgents()
        actions = gameState.getLegalActions()
        depth = 0
        agentIndex = 0

        def value(gameState, depth, agentIndex):
            agentIndex %= totalAgents
            if depth == self.depth * totalAgents: 
                return ("", self.evaluationFunction(gameState))
            if agentIndex == 0:
                return maxValue(gameState, depth, agentIndex)
            else:
                return minValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth, agentIndex):
            valueTuple = ("", float("-inf"))
            actions = gameState.getLegalActions(0)
            if len(actions) == 0:
                return ("", self.evaluationFunction(gameState))

            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                result = value(successor, depth + 1, agentIndex + 1)
                tupleVal = valueTuple[1]
                if result[1] > tupleVal:
                    valueTuple = (action, result[1])

            return valueTuple

        def minValue(gameState, depth, agentIndex):
            valueTuple = ("", float("inf"))
            actions = gameState.getLegalActions(agentIndex)
            if len(actions) == 0:
                return ("", self.evaluationFunction(gameState))

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                result = value(successor, depth + 1, agentIndex + 1)
                tupleVal = valueTuple[1]
                if result[1] < tupleVal:
                    valueTuple = (action, result[1])
            return valueTuple

        value_action = value(gameState, depth, agentIndex)
        action = value_action[0]
        return action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentIndex = 0
        value_action = self.value(gameState, depth, agentIndex)
        action = value_action[1]

        return action


    def value(self, gameState, depth, agentIndex):
    	# TODO: assign agent #'s and update depth, what exactly are we returning -> action, make tuple
    	if agentIndex == 0 or agentIndex == gameState.getNumAgents():
    		agentIndex = 0

    	if self.depth == depth:
    		return (self.evaluationFunction(gameState), "")

    	if agentIndex == 0:
    		value_action = self.max_value(gameState, depth + 1, agentIndex)

    	else:
    		value_action = self.exp_value(gameState, depth + 1, agentIndex)

    	return value_action 


    def max_value(self, gameState, depth, agentIndex):
    	value_action = (-1 * float("inf"), "")
    	# get successors
    	actions = gameState.getLegalActions(agentIndex)
    	for action in actions:
    		successor = gameState.generateSuccessor(agentIndex, action)
    		valuetuple_successor = self.value(successor, depth, agentIndex + 1)
    		value_successor = valuetuple_successor[0]

    		if value_successor > value_action[0]:
    			value_action = (value_successor, action)
    		# value_action = max(v, value(self, successor, depth, agentIndex))

    	return value_action

    def exp_value(self, gameState, depth, agentIndex):
    	value = 0
    	actions = gameState.getLegalActions(agentIndex)
    	if len(actions) == 0:
    		return (self.evaluationFunction(gameState), "")
    	probability = 1.0/float(len(actions))

    	for action in actions:
    		successor = gameState.generateSuccessor(agentIndex, action)
	    	valuetuple_successor = self.value(successor, depth, agentIndex + 1) # contains val and action
	    	value_successor = valuetuple_successor[0]
	    	value = value + probability * value_successor

    	return (value, "")

    def min_value(self, gameState, depth, agentIndex):
        value_action = (float("inf"), "")

        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, actions)
            valuetuple_successor = self.value(successor, depth, agentIndex + 1)
            value_successor = valuetuple_successor[0]

            if value_successor < value_action[0]:
                value_action = (value_successor, action)
        return value_action
    	

def calculateDistances(pacman, items):
    distances = []
    for item in items:
        distances.append(util.manhattanDistance(pacman, item))
    return distances

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    radiusSum = 0
    proximityBonus = 0
    runawayBonus = 0
    total = 0

    if (currentGameState.isWin()):
        return float("inf")

    pacmanPos = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    foodPositions = currentGameState.getFood().asList()
    pelletPositions = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    currentScore = currentGameState.getScore()
    wallPositions = currentGameState.getWalls()
    wallList = wallPositions.asList()
    total += currentScore

    gridFirstHalf = (wallPositions.width / 2)
    gridSecondHalf = (wallPositions.width / 2) + 1


    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # print newScaredTimes

    pelletDistances = calculateDistances(pacmanPos, pelletPositions)
    ghostDistances = calculateDistances(pacmanPos, ghostPositions)
    foodDistances = calculateDistances(pacmanPos, foodPositions)
    
    closestPellet = 0
    scaredtimes = sum(newScaredTimes)

    if (len(pelletDistances) != 0):
        closestPellet = min(pelletDistances)
        closestPellet = 1.0/closestPellet
        # if (scaredtimes == 0): closestPellet = 0

    closestFood, closestGhost = min(foodDistances), min(ghostDistances)


    # ## Make scared ghost more favorable to go to
    if (sum(newScaredTimes) > 5):
        runawayBonus = sum(newScaredTimes) + 10
        closestGhost = 1

    if (foodDistances != 0): 
        closestFood = 1.0/closestFood
        total += closestFood * closestGhost

    if (closestGhost > 3):
        total += 50
    if (closestGhost < 4 and closestGhost > 3 and closestPellet < 3 and scaredtimes == 0):
        total += 10

    # return total

    return currentGameState.getScore() + ((closestPellet + closestFood) * closestGhost) + radiusSum + scaredtimes + (proximityBonus + runawayBonus)


# Abbreviation
better = betterEvaluationFunction
















