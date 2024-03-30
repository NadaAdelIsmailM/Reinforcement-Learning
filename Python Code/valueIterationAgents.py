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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            states = self.mdp.getStates()
            UpdatedQ = {}
            for state in states:
                PossibleActions = self.mdp.getPossibleActions(state)
                maxval = 0
                if PossibleActions:
                    values = [self.computeQValueFromValues(state, action) for action in PossibleActions]
                    maxval = max(values)
                UpdatedQ[state] = maxval
            for state in states:
                self.values[state] = UpdatedQ[state]


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
        Q = 0
        gamma=self.discount
        for nextstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # print(nextState, Q, prob)
            sprime = self.getValue(nextstate)
            Q= Q +(prob * (self.mdp.getReward(state, action, nextstate) + gamma * sprime))
        return Q
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

        """

        "*** YOUR CODE HERE ***"
        #computes the best action according to the value function given by self.values.
        # PossibleActions = self.mdp.getPossibleActions(state)
        # if len(PossibleActions)==0:
        #     return None
        # max_value = 0
        # max_index = 0
        # for i in range(len(PossibleActions)):
        #     action = PossibleActions[i]
        #     Qvalue = self.computeQValueFromValues(state, action)
        #     if Qvalue>max_value:
        #         max_value = Qvalue
        #         max_index = i
        # myaction=PossibleActions[max_index]
        # #implement random choice for equal values
        # #Qvalues = [self.computeQValueFromValues(state, action) for action in PossibleActions]
        # #maxval = max(Qvalues)
        # #maxind = [i for i in range(len(Qvalues)) if Qvalues[i] is maxval]
        # #myaction = PossibleActions[random.choice(maxind)]
        #
        # return myaction
        #util.raiseNotDefined()
        PossibleActions = self.mdp.getPossibleActions(state)

        if len(PossibleActions) == 0:
            return None

        Qvals = []
        for action in PossibleActions:
            qval = self.getQValue(state, action)
            Qvals.append((action, qval))

        bestAction = max(Qvals, key=lambda x: x[1])[0]
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        states = self.mdp.getStates()
        fringe = util.PriorityQueue()
        predecessors = {}

        for state in states:
            self.values[state] = 0
            predecessors[state] = self.get_predecessors(state)

        for state in states:
            if not self.mdp.isTerminal(state):
                stateval = self.values[state]
                diff = abs(stateval - self.maxQ(state))
                fringe.push(state, -diff)

        for _ in range(self.iterations):

            if fringe.isEmpty():
                return

            state = fringe.pop()
            self.values[state] = self.maxQ(state)

            for p in predecessors[state]:
                diff = abs(self.values[p] - self.maxQ(p))
                if diff > self.theta:
                    fringe.update(p, -diff)


    def maxQ(self, state):
        return max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])


    # First, we define the predecessors of a state s as all states that have
    # a nonzero probability of reaching s by taking some action a
    # This means no Terminal states and T > 0.
    def get_predecessors(self, state):
        predecessor_set = set()
        states = self.mdp.getStates()
        movements = ['north', 'south', 'east', 'west']

        if not self.mdp.isTerminal(state):

            for stat in states:
                terminal = self.mdp.isTerminal(stat)
                PossibleActions = self.mdp.getPossibleActions(stat)

                if not terminal:

                    for move in movements:

                        if move in PossibleActions:
                            transition = self.mdp.getTransitionStatesAndProbs(stat, move)

                            for s_prime, T in transition:
                                if (s_prime == state) and (T > 0):
                                    predecessor_set.add(stat)

        return predecessor_set