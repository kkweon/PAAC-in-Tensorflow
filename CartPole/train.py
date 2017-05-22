import tensorflow as tf
import numpy as np
import gym
from typing import Iterable


def flag_parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon",
                        type=float,
                        default=1e-7,
                        help="Epsilon for numeric stability")

    parser.add_argument("--n-hidden",
                        type=int,
                        default=256,
                        help="Hiden unints for network")

    parser.add_argument("--entropy",
                        type=float,
                        default=0.01,
                        help="Entropy coefficient")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="RMSProp Learning rate")

    parser.add_argument("--decay",
                        type=float,
                        default=0.99,
                        help="Decay rate (used for RMSProp and Discounted Rate)")

    parser.add_argument("--gradient_clip",
                        type=float,
                        default=40,
                        help="Gradient Clip by Value (-value, value)")
    return parser.parse_args()


FLAGS = flag_parse()


class Agent(object):

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_network(self.input_dim, self.output_dim)

    def __build_network(self, input_dim, output_dim):

        self.states = tf.placeholder(tf.float32, shape=[None, input_dim], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        action_onehot = tf.one_hot(self.actions, depth=output_dim, name="action_onehot")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        net = tf.layers.dense(self.states,
                              units=FLAGS.n_hidden,
                              activation=tf.nn.relu,
                              name="fc1")
        net = tf.layers.dense(net,
                              units=FLAGS.n_hidden,
                              activation=tf.nn.relu,
                              name="fc2")

        action_scores = tf.layers.dense(net, units=output_dim, name="action_scores")
        self.action_probs = tf.nn.softmax(action_scores, name="action_probs")

        single_action_prob = tf.reduce_sum(self.action_probs * action_onehot, axis=1)
        log_action_prob = tf.log(single_action_prob + FLAGS.epsilon)

        entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + FLAGS.epsilon), axis=1)

        self.actor_loss = - tf.reduce_mean(log_action_prob * self.advantages + entropy * FLAGS.entropy)

        self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.values, self.rewards))

        self.optim = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate,
                                               decay=FLAGS.decay)

        self.total_loss = self.actor_loss + self.value_loss

        self.global_step = tf.train.get_or_create_global_step()

        gradients = self.optim.compute_gradients(self.total_loss)
        gradients = [(tf.clip_by_value(grad, -FLAGS.gradient_clip, FLAGS.gradient_clip), var) for grad, var in gradients]
        self.train_op = self.optim.apply_gradients(gradients, global_step=self.global_step)

    def get_actions(self, states):
        """Returns an action

        Always returns an array of shape (N,) because there are N environments

        Args:
            states (1-D or 2-D Tensor): It could be a tensor of shape (N, D) or (D,)

        Returns:
            action (1-D Tensor): Action array, (N,)
        """

        sess = tf.get_default_session()
        states = np.reshape(states, (-1, self.input_dim))

        feed = {
            self.states: states
        }

        action_probs = sess.run(self.action_probs, feed)

        noises = np.random.uniform(size=action_probs.shape[0]).reshape(-1, 1)
        actions = (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1)

        return actions

    def get_values(self, states):
        """Returns multiple values

        Always returns an array of shape (N,) because there are N environments

        Args:
            states (1-D or 2-D Tensor): It could be a tensor of shape (N, D) or (D,)

        Returns:
            values (1-D Tensor): Value array, (N,)
        """
        sess = tf.get_default_session()
        states = np.reshape(states, (-1, self.input_dim))

        feed = {
            self.states: states
        }

        values = sess.run(self.values, feed)

        return values.reshape(-1)

    def train(self, states, actions, rewards, values):
        """Train multiple environments

        Args:
            states (3-D Tensor): (n_env, n_step, input_dim)
            actions (2-D Tensor): (n_env, n_step)
            rewards (2-D Tensor): (n_env, n_step)
            values (2-D Tensor): (n_env, n_step)
        """
        states = np.vstack(list(filter(lambda x: len(x) > 0, states)))
        actions = np.hstack(actions)

        rewards = discounted_multienv_rewards(rewards, gamma=FLAGS.decay)
        rewards = np.hstack(rewards)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + FLAGS.epsilon

        values = np.hstack(values)

        advantages = rewards - values
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + FLAGS.epsilon

        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.advantages: advantages
        }

        sess = tf.get_default_session()
        sess.run(self.train_op, feed)


def train_episodes(envs: Iterable[gym.Env], agent: Agent, t_max: int, pipeline=lambda x: x):
    """
    Args:
        envs (list): Each element is a `gym.Env`
        agent (Agent): A single agent
        t_max (int): Number of steps before update
    """
    n_envs = len(envs)

    state_memory = [[] for env in envs]
    action_memory = [[] for env in envs]
    reward_memory = [[] for env in envs]
    value_memory = [[] for env in envs]

    all_envs_done = False

    env_is_done = [False for env in envs]
    observations = [pipeline(env.reset()) for env in envs]
    env_total_reward = [0 for env in envs]

    while not all_envs_done:

        for _ in range(t_max):

            actions = agent.get_actions(observations)
            values = agent.get_values(observations)

            for id, env in enumerate(envs):

                if not env_is_done[id]:
                    s2, r, env_is_done[id], _ = env.step(actions[id])

                    state_memory[id].append(observations[id])
                    action_memory[id].append(actions[id])
                    reward_memory[id].append(r)
                    value_memory[id].append(values[id])

                    env_total_reward[id] += r

                    observations[id] = pipeline(s2)

        extra_values = agent.get_values(observations)

        for id in range(n_envs):

            if not env_is_done[id]:

                reward_memory[id][-1] += FLAGS.decay * extra_values[id]

        agent.train(state_memory, action_memory, reward_memory, value_memory)
        state_memory = [[] for env in envs]
        action_memory = [[] for env in envs]
        reward_memory = [[] for env in envs]
        value_memory = [[] for env in envs]

        all_envs_done = np.all(env_is_done)

    return env_total_reward


def discounted_multienv_rewards(multienv_rewards, gamma=FLAGS.decay):
    """
    Args:
        multienv_rewards (2-D Tensor): Rewards (n_env, n_step)

    Returns:
        discounted_rewards (2-D Tensor): Discounted Rewards (n_env, n_step)
    """
    N_env = len(multienv_rewards)

    for i in range(N_env):
        multienv_rewards[i] = discount_rewards(multienv_rewards[i], gamma)

    return multienv_rewards


def discount_rewards(rewards, gamma=FLAGS.decay):
    discounted = np.zeros_like(rewards, dtype=np.float32)

    running_add = 0
    for i in reversed(range(len(rewards))):

        running_add = rewards[i] + gamma * running_add
        discounted[i] = running_add

    return discounted


def get_env_info(env_id: str):
    try:
        test_env = gym.make(env_id)
        input_dim = test_env.observation_space.shape[0]
        output_dim = test_env.action_space.n

    finally:
        test_env.close()

    return input_dim, output_dim


def main():
    env_id = "CartPole-v0"
    input_dim, output_dim = get_env_info(env_id)
    n_envs = 32
    monitor_dir = "monitor"

    try:
        agent = Agent(input_dim, output_dim)
        envs = [gym.make(env_id) for _ in range(n_envs)]

        envs[0] = gym.wrappers.Monitor(envs[0], monitor_dir, force=True)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()

            sess.run(init)

            i = 0
            while True:
                env_rewards = train_episodes(envs, agent, 5)
                print(i + 1, np.mean(env_rewards), env_rewards)

                i += 1
    finally:
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
