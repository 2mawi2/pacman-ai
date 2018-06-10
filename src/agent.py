import math

import tensorflow as tf
import numpy as np
import random


class Agent:
    def __init__(self, n_actions, n_states, discount=0.8, alpha=0.1, epsilon=1, epsilon_decay=0.99, lambda_=0.5):
        self.n_actions = n_actions
        self.n_states = n_states
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.backup_epsilon = self.epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_

        tf.reset_default_graph()
        self.state_tensor, self.Q_values_tensor, self.chosen_value_tensor, self.opt, self.weight1 = self._build_model()
        self.grads_and_vars = self._get_gradients(self.opt)
        self.e_trace = self._get_eligibility_trace(self.grads_and_vars)

        self.grad_placeholder = [(tf.placeholder("float", shape=grad[0].get_shape()), grad[1]) for grad in
                                 self.grads_and_vars]
        self.apply_placeholder_op = self.opt.apply_gradients(self.grad_placeholder)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        state_tensor = tf.placeholder(tf.float32, shape=(1, self.n_states))
        # weight1 = tf.Variable(tf.truncated_normal(shape=(self.n_states, self.n_actions), stddev=0.01))
        # bias1 = tf.Variable(tf.zeros(self.n_actions))
        # q_values_tensor = tf.add(tf.matmul(state_tensor, weight1), bias1)

        weight1 = tf.Variable(tf.zeros(shape=(self.n_states, self.n_actions)))
        q_values_tensor = tf.matmul(state_tensor, weight1)
        chosen_action_index = tf.argmax(q_values_tensor, 1)
        chosen_value_tensor = tf.gather(q_values_tensor, chosen_action_index, axis=1)
        opt = tf.train.GradientDescentOptimizer(self.alpha)

        return state_tensor, q_values_tensor, chosen_value_tensor, opt, weight1

    def _get_gradients(self, opt):
        """
        opt.compute_gradients returns a list of gradients (of 'self.chosen_value_tensor')
        and the variables they correspond to, with respect to 'trainable_variables'
        """
        trainable_variables = tf.trainable_variables()
        return opt.compute_gradients(self.chosen_value_tensor, trainable_variables)

    def _get_eligibility_trace(self, grads_and_vars):
        e_trace = []
        for gv in grads_and_vars:
            e = np.zeros(gv[0].get_shape())
            e_trace.append(e)
        return e_trace

    def _compute_e_trace(self, evaluated_gradients, e_trace):
        for i in range(len(e_trace)):
            e_trace[i] = self.discount * self.lambda_ * e_trace[i] + evaluated_gradients[i]
            assert (e_trace[i].shape == evaluated_gradients[i].shape)
        return e_trace

    def predict_Q_values(self, state):
        return self.sess.run(self.Q_values_tensor, feed_dict={self.state_tensor: state})

    def get_max_Q_value(self, state):
        return np.max(self.predict_Q_values(state))

    def get_Q_value(self, state, action):
        return self.predict_Q_values(state)[0][action]

    def get_best_action(self, state):
        return np.argmax(self.predict_Q_values(state))

    def get_e_greedy_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.n_actions)
            if action == self.get_best_action:
                return action, False
            else:
                return action, True
        else:
            return self.get_best_action(state), False

    def reset_e_trace(self):
        self.e_trace = [0 * e for e in self.e_trace]

    def print_weights(self):
        w1 = self.get_weights()
        print(w1)

    def get_weights(self):
        return self.sess.run(self.weight1)

    def print_Q_values(self, state):
        print(self.predict_Q_values(state))

    def learn(self, state, action, next_state, reward, greedy):
        delta = reward + self.get_max_Q_value(next_state) - self.get_Q_value(state, action)

        assert abs(delta) < 1000

        # as per Watkin's Q, if the target policy wouldn't have produced the same action, the trace is set to 0
        if greedy:
            self.reset_e_trace()
        else:
            evaluated_gradients = self.get_gradients(state)
            self.e_trace = self._compute_e_trace(evaluated_gradients, self.e_trace)

        change = self.add_negative_sign(delta)
        self.apply_gradient_update(change)
        self.decay_epsilon()

    def get_gradients(self, state):
        """Getting gradients (and the variables they correspond to)"""
        grads_and_vars = self.sess.run(self.grads_and_vars, feed_dict={self.state_tensor: state})
        return [gv[0] for gv in grads_and_vars]

    def add_negative_sign(self, delta):
        """since tensorflow tries to minimize"""
        return [-delta * e for e in self.e_trace]

    def apply_gradient_update(self, change):
        """
        Eligibility trace (e_trace) is essentially a modified gradient.change is the change to be applied to the weigth
        To alter the gradients before applying them, we have to do some session running and dictionary feeding
        """
        feed_dict = {}
        for i in range(len(self.grad_placeholder)):
            feed_dict[self.grad_placeholder[i][0]] = change[i]
        self.sess.run(self.apply_placeholder_op, feed_dict=feed_dict)

    def decay_epsilon(self):
        """
           this should be improved as VDBE-Boltzmann with an adaptive e-greedy exploration rate
           http://tokic.com/www/tokicm/publikationen/papers/AdaptiveEpsilonGreedyExploration.pdf
        """
        self.epsilon = (self.epsilon * self.epsilon_decay)
