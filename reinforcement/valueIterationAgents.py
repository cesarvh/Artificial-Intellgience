# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
from util import PriorityQueue

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        iterationCounter = 0 
        while iterationCounter != self.iterations:
            vals = self.values.copy()
            for state in self.mdp.getStates():
                actionValues = []
                for action in self.mdp.getPossibleActions(state):
                    actionVal = 0
                    for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        actionVal += probability * (self.mdp.getReward(state, action, transition) + self.discount * vals[transition])
                    actionValues.append(actionVal)
                self.values[state] = max(actionValues or [0])
            iterationCounter += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        qValue = 0
        for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += probability * (self.mdp.getReward(state, action, transition) + self.discount * self.values[transition])
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        counter = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            counter[action] += self.computeQValueFromValues(state, action)
        return counter.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.gtPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # "*** YOUR CODE HERE ***"
        # states = self.mdp.getStates()
        # iterationCounter = 0 
        # i = 0
        # while iterationCounter < self.iterations:
        #     i = iterationCounter % len(states)
        #     # print("i is ", i)
        #     # print("iterationCounter is ", iterationCounter)

        #     state = states[i]
        #     if self.mdp.isTerminal(state) == False:
        #         for action in self.mdp.getPossibleActions(state):
        #             actionValues = []
        #             actionVal = 0
        #             for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
        #                 reward = self.mdp.getReward(state, action, transition)
        #                 actionVal += probability * (reward + self.discount * self.values[transition])
        #             actionValues.append(actionVal)

        #         # self.values[state] = max(actionValues or self.values[state])
        #         # print("self.values ", self.values[state])
        #         # print("max actionvals ", max(actionValues))
        #         if self.values[state] < max(actionValues):
        #             self.values[state] = max(actionValues)

        #     i += 1
        #     iterationCounter += 1
        iterationCounter = 0 
        states = self.mdp.getStates()
        while iterationCounter != self.iterations:
            vals = self.values.copy()
            state = states[iterationCounter % len(states)]
            actionValues = []
            for action in self.mdp.getPossibleActions(state):
                actionVal = 0
                for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    actionVal += probability * (self.mdp.getReward(state, action, transition) + self.discount * vals[transition])
                actionValues.append(actionVal)
            self.values[state] = max(actionValues or [0])
            iterationCounter += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = PriorityQueue()
        states = self.mdp.getStates()
        predecessors = dict()

        # compute predecessors
        for state in states:
            # predecessors[state] = set()
            for action in self.mdp.getPossibleActions(state):
                for transition, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    if probability != 0:
                        if transition in predecessors:
                            predecessors[transition].add(state)
                        else:
                            predecessor_set = set()
                            predecessor_set.add(state)
                            predecessors[transition] = predecessor_set

        for state in states:
            if self.mdp.isTerminal(state) == False:
                curr = self.values[state]
                actionValues = []
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    actionValues.append(qValue)
                highest_q = max(actionValues)
                diff = abs(curr - highest_q)
                pq.push(state, -1 * diff)

        i = 0
        while i != self.iterations:
            if pq.isEmpty():
                return
            state_s = pq.pop()

                #oops 
            actionValues = []
            actions = self.mdp.getPossibleActions(state_s)
            if actions:
                for action in self.mdp.getPossibleActions(state_s):
                    qValue = self.computeQValueFromValues(state_s, action)
                    actionValues.append(qValue)
                self.values[state_s] = max(actionValues)

            predecessors_s = predecessors[state_s]
            for predecessor in predecessors_s:
            	actionValues = []
                for action in self.mdp.getPossibleActions(predecessor):
                    qValue = self.computeQValueFromValues(predecessor, action)
                    actionValues.append(qValue)
                highest_q = max(actionValues)
                diff = abs(self.values[predecessor] - highest_q)
                if diff > self.theta:
                    pq.update(predecessor, -1 * diff)

            i += 1



        # pq = PriorityQueue()
        # states = self.mdp.getStates()
        # predecessors = dict()

        # #compute predecessors
        # for s in states:
        #     possibleActions = self.mdp.getPossibleActions(s)
        #     for action in possibleActions:
        #         for transition, probability in self.mdp.getTransitionStatesAndProbs(s, action):
        #             if probability != 0:
        #                 if transition in predecessors:
        #                     predecessors[transition].add(s)
        #                 else:
        #                     predecessors[transition] = set(s)


        # for s in states:
        #     if self.mdp.isTerminal(s) == False:
        #         possibleActions = self.mdp.getPossibleActions(s)
        #         actionValues = []
        #         for action in possibleActions:
        #             qValue = self.computeQValueFromValues(s, action)
        #             actionValues.append(qValue)
        #         highest_q = max(actionValues)
        #         diff = abs(highest_q - self.values[s])
        #         pq.push(s, -diff)


        # i = 0
        # while i != self.iterations:
        #     if pq.isEmpty():
        #         return
        #     s = pq.pop()
        #     if self.mdp.isTerminal(s) == False:
        #         possibleActions = self.mdp.getPossibleActions(s)
        #         actionValues = []
        #         for action in possibleActions:
        #             qValue = self.computeQValueFromValues(s, action)
        #             actionValues.append(qValue)
        #         self.values[s] = max(actionValues or [0])

        #     predecessors_s = predecessors[s]
        #     for p in predecessors_s:
        #         possibleActions_p = self.mdp.getPossibleActions(p)
        #         actionValues = []
        #         for action in possibleActions_p:
        #             qValue = self.computeQValueFromValues(p, action)
        #             actionValues.append(qValue)
        #         highest_q = max(actionValues or [0])
        #         diff = abs(highest_q - self.values[p])

        #         if diff > self.theta:
        #             pq.update(p, -diff)

        #     i += 1




