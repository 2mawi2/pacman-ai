#!/usr/local/bin/python


import random

# Instead of letters, I use number for the states. The states 0 and 6 are the final states.
# The states 1,...,5 are the states A,...,E

states = range(0, 7)
finalStates = [0, 6]
reward = [1 if s == 6 else 0 for s in states]


def policy(s):
    """Random walk policy: ations are 'go left' nand 'go right'."""

    return random.choice(['left', 'right'])


def execute_policy(s, a):
    """Change state based on the taken action."""

    if a == 'left':
        return s - 1
    else:
        return s + 1


def TD_0(V_star, alpha, gamma, numOfEpisodes=10000):
    """Use Temporal-Difference Learning to learn V^*."""

    for episode in range(numOfEpisodes):
        # select random start state
        s = random.randint(min(states) + 1, max(states) - 1)

        endOfEpisode = False
        while not endOfEpisode:
            if s in finalStates:
                # evaluate the value of the final state
                V_star[s] = V_star[s] + alpha * (reward[s] - V_star[s])

                # because we are in the final state then end the episode
                endOfEpisode = True
                continue
            else:
                # get an action for this state from the policy
                a = policy(s)

                # execute policy => take an action
                s_prime = execute_policy(s, a)

                # evaluate the action
                V_star[s] = V_star[s] + alpha * (reward[s] + gamma * V_star[s_prime] - V_star[s])

                s = s_prime


def V(s, d=0):
    """Value function computed by dynamic programing."""

    if d > 20:
        return 0

    if s in finalStates:
        return reward[s]

    return 0.5 * (V(s - 1, d + 1) + V(s + 1, d + 1))


###############################################################################
##
##  Experiments
##
###############################################################################

gamma = 1

print("TD(0): alpha = 0.15")

# init the value function
V_star = [0.5 for s in states]
TD_0(V_star, 0.15, gamma, 100000)
for s, V_s in enumerate(V_star):
    V_s_star = s / 6.0
    print("V(%d) = %0.3f   err = %.3f" % (s, V_s, abs(V_s_star - V_s)))

print("TD(0): alpha = 0.05")

# init the value function
V_star = [0.5 for s in states]
TD_0(V_star, 0.05, gamma, 100000)
for s, V_s in enumerate(V_star):
    V_s_star = s / 6.0
    print("V(%d) = %0.3f   err = %.3f" % (s, V_s, abs(V_s_star - V_s)))

print
"Dynamic programing:"
for s in states:
    V_s_star = s / 6.0
    V_s = V(s)
    print("V(%d) = %0.3f   err = %.3f" % (s, V_s, abs(V_s_star - V_s)))
