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
        depth = 0
        agentIndex = 0
        value_action = self.value(gameState, depth, agentIndex)
        action = value_action[1]

        return action

    def value(self, gameState, depth, agentIndex):
        # TODO: assign agent #'s and update depth, what exactly are we returning -> action, make tuple
        if agentIndex == 0 or agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        # if gameState.isWin() or gameState.isLose():
        #     return gameState.getScore()

        if self.depth == depth - 1 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), "")

        if agentIndex == 0:
            value_action = self.max_value(gameState, depth, agentIndex)

        else:
            value_action = self.min_value(gameState, depth, agentIndex)

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

    def min_value(self, gameState, depth, agentIndex):
        value_action = (float("inf"), "")
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            valuetuple_successor = self.value(successor, depth, agentIndex + 1)
            value_successor = valuetuple_successor[0]

            if value_successor < value_action[0]:
                value_action = (value_successor, action)
            # value_action = max(v, value(self, successor, depth, agentIndex))

        return value_action

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentIndex = 0
        alpha = float("-inf")
        beta = float("inf")
        value_action = self.value(gameState, depth, agentIndex, alpha, beta)
        action = value_action[1]

        return action

    def value(self, gameState, depth, agentIndex, alpha, beta):
        # TODO: assign agent #'s and update depth, what exactly are we returning -> action, make tuple
        if agentIndex == 0 or agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        # if gameState.isWin() or gameState.isLose():
        #     return gameState.getScore()

        if self.depth == depth - 1 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), "")

        if agentIndex == 0:
            value_action = self.max_value(gameState, depth, agentIndex, alpha, beta)

        else:
            value_action = self.min_value(gameState, depth, agentIndex, alpha, beta)

        return value_action 

    def max_value(self, gameState, depth, agentIndex, alpha, beta):
        value_action = (-1 * float("inf"), "")
        # get successors
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            valuetuple_successor = self.value(successor, depth, agentIndex + 1, alpha, beta)
            value_successor = valuetuple_successor[0]

            if value_successor > value_action[0]:
                value_action = (value_successor, action)

            if value_action[0] > beta:
                return value_action

            alpha = max(alpha, value_successor)

        return value_action

    def min_value(self, gameState, depth, agentIndex, alpha, beta):
        value_action = (float("inf"), "")
        # get successors
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            valuetuple_successor = self.value(successor, depth, agentIndex + 1, alpha, beta)
            value_successor = valuetuple_successor[0]

            if value_successor < value_action[0]:
                value_action = (value_successor, action)

            if value_action[0] < alpha:
                return value_action

            beta = min(beta, value_successor)

        return value_action

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
            depth += 1

        if self.depth == depth - 1 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), "")

        if agentIndex == 0:
            value_action = self.max_value(gameState, depth, agentIndex)

        else:
            value_action = self.exp_value(gameState, depth, agentIndex)

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
    # Initialize things that MAY later change to 0.
    radiusSum = 0 
    runawayBonus = 0
    closestPellet = 0

    # If we reached the last food pellet, we want to end the game immediately
    # So we send an infinite # so we know the state is good
    if (currentGameState.isWin()):
        return float("inf")

    # Now we want to get the current positions of the Pacman, Ghosts, Food and Pellets
    # In order to calculate stuff. Note FoodPositions is a list of tuples of integers
    pacmanPos = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    foodPositions = currentGameState.getFood().asList()
    pelletPositions = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()

    # This gets the amount of time left that the ghost will be scared/eatable
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    scaredtimes = sum(newScaredTimes) # We call sum() to get an integer from the array. Equal to the time left scared

    # Now we will compute the distance to the foods, the ghosts, and the power pellets.
    # These are LISTS of distances from pacman --> item
    pelletDistances = calculateDistances(pacmanPos, pelletPositions)
    ghostDistances = calculateDistances(pacmanPos, ghostPositions)
    foodDistances = calculateDistances(pacmanPos, foodPositions)
    
    # If there are still pellets left on the board
    if (len(pelletDistances) != 0):
        closestPellet = min(pelletDistances) # Then we get the distance to the closest pellet
        closestPellet = 1.0/closestPellet - 0.09 
        # We set closestPellet equal to its receiprocal because the spec recommended it. We also subtract .09 because
        # If we don't, pellets and food will be weighted equally, which will cause pacman to prefer getting pellets over food
        # This caused pacman to stop moving until the ghost came by to eat it.

    # Get the closest food and the closest ghost from the arrays calculated above
    closestFood, closestGhost = min(foodDistances), min(ghostDistances)
    closestFood = 1.0/closestFood # We use the reciprocal idk why but the spec reccomended it LOL


    # Make scared ghost more favorable to go to
    # If we ghost is cared for more than 5 seconds, we make it so that pacman will be
    # More likely to go to that ghost, that way he can eat it.
    if (sum(newScaredTimes) > 5):
        runawayBonus = sum(newScaredTimes) + 15 # Here we set the weight that will make pacman more likely go to a ghost
        closestGhost = 1 # Since we're not scared of ghosts, we set this variable = 1 because we're not scared of ghosts 
                         # Anymore. Note we can't Let it equal 0 because otherwise that would mess up the pellet and food weights later

    # Okay, now, if the ghost isn't within 3 Manhattan Distance units from us, then essentially we should feel free to move
    # This should make pacman like this state more, so we increase radiusSum to increase the weight of this state
    if (closestGhost > 3):
        radiusSum = 10
    # If closest Ghost is further away, we make it even more likely for pacman to go there. 
    if closestGhost > 10:
        radiusSum = 50

    # Here, I took the current game score because the higher the score, the more we want to go to that state
    # I also added the closest pellet because want Pacman to get the pellet when possible
    # I then multiplied the closest food to the closest ghost idk why tbh but it worked LOL. If i divide, it doesn't. If I add, it doesnt :/
    # I also added the radius sum (see line 437 -441). It's 0 if its not favorable.
    # Added scared times, because if the ghost is scared, we shouldn't worry about it.
    # Added Runaway bonus, see lines 431-433
    return currentGameState.getScore() + closestPellet + (closestFood * closestGhost) + radiusSum + scaredtimes  + runawayBonus

# Abbreviation
better = betterEvaluationFunction

