"""Summary

Attributes:
    parser (TYPE): Description
"""
import argparse
import tensorflow as tf
import numpy as np
import gym
from scipy.misc import imresize
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon",
                    type=float,
                    default=1e-8,
                    help="Epsilon value for numeric stability")

parser.add_argument("--decay",
                    type=float,
                    default=.99,
                    help="Decay rate for RMSProp and Discount Rate")

parser.add_argument("--learning-rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for RMSProp")

parser.add_argument("--norm",
                    type=float,
                    default=50,
                    help="Gradient clip by its norm value")

parser.add_argument("--entropy",
                    type=float,
                    default=0.05,
                    help="Entropy coefficient")

parser.add_argument("--t-max",
                    type=int,
                    default=10,
                    help="Update period")

parser.add_argument("--n-envs",
                    type=int,
                    default=16,
                    help="Number of parallel environments")

FLAGS, _ = parser.parse_known_args()


def resize_image(image, new_HW=(40, 40)):
    """Returns a resize image

    Args:
        image (3-D Array): RGB Image Array of shape (H, W, C)
        new_HW (tuple, optional): New Height and Width (height, width)

    Returns:
        3-D Array: A resized image of shape (`height`, `width`, C)
    """
    return imresize(image, new_HW)


def crop_ROI(image, height_range=(35, 210), width_range=(8, 150)):
    """Crops a region of interest (ROI)

    Args:
        image (3-D Array): RGB Image of shape (H, W, C)
        height_range (tuple, optional): Height range to keep (h_begin, h_end)
        width_range (tuple, optional): Width range to keep (w_begin, w_end)

    Returns:
        3-D array: Cropped image of shape (h_end - h_begin, w_end - w_begin, C)
    """
    h_beg, h_end = height_range
    w_beg, w_end = width_range
    return image[h_beg:h_end, w_beg:w_end, ...]


def binarize_image(image):
    """Binarizes image (make everything 0 or 1)

    Args:
        image (3-D Array): RGB Image of shape (H, W, C)

    Returns:
        2-D Array: Binarized image of shape (H, W)
    """
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R > 0) & (G > 0) & (B > 0)

    binarized = np.zeros_like(R)
    binarized[cond] = 1

    return binarized


def pipeline(image, new_HW=(40, 40)):
    """Image process pipeline

    Args:
        image (3-D Array): 3-D array of shape (H, W, C)

    Returns:
        3-D Array: Binarized image of shape (H, W, 1)
    """
    image = crop_ROI(image)
    image = resize_image(image, new_HW=new_HW)

    return np.expand_dims(binarize_image(image), axis=2)


def discount_rewards(rewards, gamma=FLAGS.decay):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0

    for i in reversed(range(len(rewards))):
        if rewards[i] < 0:
            running_add = 0
        running_add = rewards[i] + gamma * running_add
        discounted[i] = running_add
    return discounted


def discount_multi_rewards(multi_rewards, gamma=FLAGS.decay):
    """
    Args:
        multi_rewards (2-D Array): Reward array of shape (n_envs, n_timesteps)
        gamma (float, optional): Discount rate for a reward

    Returns:
        discounted_multi_rewards (2-D Array): Reward array of shape (n_envs, n_timesteps)
    """
    n_envs = len(multi_rewards)
    discounted = []
    for id in range(n_envs):
        discounted.append(discount_rewards(multi_rewards[id], gamma))
    return discounted


class Agent(object):
    def __init__(self, input_shape, output_dim):
        """Summary

        Args:
            input_shape (TYPE): Description
            output_dim (TYPE): Description
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.__build_network(self.input_shape, self.output_dim)

    def __build_network(self, input_shape, output_dim):
        """Summary

        Args:
            input_shape (TYPE): Description
            output_dim (TYPE): Description
        """
        self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
        action_onehots = tf.one_hot(self.actions, depth=output_dim, name="action_onehots")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantages = tf.placeholder(tf.float32, shape=[None], name="advantages")

        net = self.states
        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net, filters=16, kernel_size=(8, 8), strides=(4, 4), name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net, filters=32, kernel_size=(4, 4), strides=(2, 2), name="conv")
            net = tf.nn.relu(net, name="relu")

        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope("fc1"):
            net = tf.layers.dense(net, units=256, name="fc")
            net = tf.nn.relu(net, name="relu")

        action_scores = tf.layers.dense(net, units=output_dim, name="action_scores")
        self.action_probs = tf.nn.softmax(action_scores, name="action_probs")

        single_action_prob = tf.reduce_sum(self.action_probs * action_onehots, axis=1)
        action_loss = - tf.log(single_action_prob + FLAGS.epsilon) * self.advantages
        entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + FLAGS.epsilon), axis=1)
        action_loss = tf.reduce_mean(action_loss - entropy * FLAGS.entropy)

        self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))

        value_loss = tf.reduce_mean(tf.squared_difference(self.values, self.rewards))

        self.loss = action_loss + value_loss

        self.optim = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate,
                                               decay=FLAGS.decay)

        gradients = self.optim.compute_gradients(self.loss)
        gradients = [(tf.clip_by_average_norm(grad, FLAGS.norm), var) for grad, var in gradients]
        global_step = tf.train.get_or_create_global_step()
        self.train_op = self.optim.apply_gradients(gradients, global_step)

    def get_actions(self, states):
        """
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)

        Returns:
            actions (1-D Array): Action Array of shape (N,)
        """
        sess = tf.get_default_session()
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        action_probs = sess.run(self.action_probs, feed)
        noises = np.random.uniform(size=action_probs.shape[0]).reshape(-1, 1)

        return (np.cumsum(action_probs) > noises).argmax(axis=1)

    def get_values(self, states):
        """
        Args:
            states (4-D Array): States Array of shape (N, H, W, C)

        Returns:
            values (1-D Array): Values (N,)
        """
        sess = tf.get_default_session()
        feed = {
            self.states: np.reshape(states, [-1, *self.input_shape])
        }
        return sess.run(self.values, feed).reshape(-1)

    def train(self, states, actions, rewards, values):
        """Update parameters by gradient descent

        Args:
            states (5-D Array): Image arrays of shape (n_envs, n_timesteps, H, W, C)
            actions (2-D Array): Action arrays of shape (n_envs, n_timesteps)
            rewards (2-D Array): Rewards array of shape (n_envs, n_timesteps)
            values (2-D Array): Value array of shape (n_envs, n_timesteps)
        """

        states = np.vstack([s for s in states if len(s) > 0])
        actions = np.hstack(actions)
        values = np.hstack(values)

        rewards = discount_multi_rewards(rewards, FLAGS.decay)
        rewards = np.hstack(rewards)

        advantages = rewards - values
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + FLAGS.epsilon

        sess = tf.get_default_session()
        feed = {
            self.states: states,
            self.actions: actions,
            self.values: values,
            self.rewards: rewards,
            self.advantages: advantages
        }
        sess.run(self.train_op, feed)


def run_episodes(envs, agent: Agent, t_max=FLAGS.t_max, pipeline=pipeline):
    n_envs = len(envs)
    all_dones = False

    states_memory = [[] for _ in range(n_envs)]
    actions_memory = [[] for _ in range(n_envs)]
    rewards_memory = [[] for _ in range(n_envs)]
    values_memory = [[] for _ in range(n_envs)]

    observations = [pipeline(env.reset()) for env in envs]
    is_env_done = [False for _ in range(n_envs)]
    lives_info = [5 for _ in range(n_envs)]
    episode_rewards = [0 for _ in range(n_envs)]

    while not all_dones:

        for t in range(t_max):

            actions = agent.get_actions(observations)
            values = agent.get_values(observations)

            for id, env in enumerate(envs):

                if not is_env_done[id]:

                    s2, r, is_env_done[id], info = env.step(actions[id])

                    episode_rewards[id] += r

                    assert type(r) == float

                    if info['ale.lives'] < lives_info[id]:
                        r = -1.0
                        lives_info[id] -= 1

                    s2 = pipeline(s2)

                    states_memory[id].append(observations[id])
                    actions_memory[id].append(actions[id])
                    rewards_memory[id].append(r)
                    values_memory[id].append(values[id])

                    observations[id] = s2

        future_values = agent.get_values(observations)

        for id in range(n_envs):
            if not is_env_done[id]:
                rewards_memory[id][-1] += FLAGS.decay * future_values[id]

        agent.train(states_memory, actions_memory, rewards_memory, values_memory)

        states_memory = [[] for _ in range(n_envs)]
        actions_memory = [[] for _ in range(n_envs)]
        rewards_memory = [[] for _ in range(n_envs)]
        values_memory = [[] for _ in range(n_envs)]

        all_dones = np.all(is_env_done)

    return episode_rewards


def main():
    GAME_ID = "Breakout-v0"
    input_shape = [40, 40, 1]
    output_dim = 4
    pipeline_fn = partial(pipeline, new_HW=input_shape[:-1])

    try:
        envs = [gym.make(GAME_ID) for i in range(FLAGS.n_envs)]
        envs[0] = gym.wrappers.Monitor(envs[0], "monitors", force=True)

        summary_writers = [tf.summary.FileWriter(logdir="logdir/env-{}".format(i)) for i in range(FLAGS.n_envs)]
        agent = Agent(input_shape, output_dim)

        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint("logdir/")

        with tf.Session() as sess:
            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
            episode = 1
            while True:
                rewards = run_episodes(envs, agent, pipeline=pipeline_fn)
                print(rewards)

                for id, r in enumerate(rewards):
                    summary = tf.Summary()
                    summary.value.add(tag="Episode Reward", simple_value=r)
                    summary_writers[id].add_summary(summary, global_step=episode)
                    summary_writers[id].flush()

                if episode % 10 == 0:
                    saver.save(sess, "logdir/model.ckpt", write_meta_graph=False)

                episode += 1

    finally:
        for env in envs:
            env.close()

        for writer in summary_writers:
            writer.close()


if __name__ == '__main__':
    main()
