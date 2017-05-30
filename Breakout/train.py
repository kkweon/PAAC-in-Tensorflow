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
from typing import Iterable
from skimage.color import rgb2gray


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon",
                    type=float,
                    default=1e-7,
                    help="Epsilon value for numeric stability")

parser.add_argument("--decay",
                    type=float,
                    default=.99,
                    help="Decay rate for AdamOptimizer and Discount Rate")

parser.add_argument("--learning-rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for AdamOptimizer")

parser.add_argument("--norm",
                    type=float,
                    default=50,
                    help="Gradient clip by its norm value")

parser.add_argument("--entropy",
                    type=float,
                    default=0.01,
                    help="Entropy coefficient")

parser.add_argument("--t-max",
                    type=int,
                    default=50,
                    help="Update period")

parser.add_argument("--n-envs",
                    type=int,
                    default=16,
                    help="Number of parallel environments")

parser.add_argument("--logdir",
                    type=str,
                    default="logdir",
                    help="Log directory")

parser.add_argument("--env",
                    type=str,
                    default="Breakout-v0",
                    help="Environment Name")

FLAGS, _ = parser.parse_known_args()


def resize_image(image, new_HW):
    """Returns a resize image

    Args:
        image (3-D Array): RGB Image Array of shape (H, W, C)
        new_HW (tuple, optional): New Height and Width (height, width)

    Returns:
        3-D Array: A resized image of shape (`height`, `width`, C)
    """
    return imresize(image, new_HW, interp='nearest')


def crop_ROI(image, height_range=(35, 210), width_range=(0, 160)):
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

    cond = (R > 0) | (G > 0) | (B > 0)

    binarized = np.zeros_like(R, dtype=np.float32)
    binarized[cond] = 1

    return binarized


def pipeline(image, new_HW):
    """Image process pipeline

    Args:
        image (3-D Array): 3-D array of shape (H, W, C)
        new_HW (tuple): New height and width int tuple of (height, width)

    Returns:
        3-D Array: Binarized image of shape (height, width, 1)
    """
    image = crop_ROI(image)
    image = resize_image(image, new_HW=new_HW)
    # image = binarize_image(image)
    image = rgb2gray(image)

    image = (image - np.mean(image)) / (np.std(image) + 1e-8)

    return np.expand_dims(image, axis=2)


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
    def __init__(self, input_shape: list, output_dim: int):
        """Summary

        Args:
            input_shape (TYPE): Description
            output_dim (TYPE): Description
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.__build_network(self.input_shape, self.output_dim)

    def __build_network(self, input_shape: list, output_dim: int):
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

        with tf.variable_scope("action_network"):
            action_scores = tf.layers.dense(net, units=output_dim, name="action_scores")
            self.action_probs = tf.nn.softmax(action_scores, name="action_probs")
            single_action_prob = tf.reduce_sum(self.action_probs * action_onehots, axis=1)
            log_action_prob = - tf.log(single_action_prob + FLAGS.epsilon) * self.advantages
            action_loss = tf.reduce_sum(log_action_prob)

        with tf.variable_scope("entropy"):
            entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + FLAGS.epsilon), axis=1)
            entropy_sum = tf.reduce_sum(entropy)

        with tf.variable_scope("value_network"):
            self.values = tf.squeeze(tf.layers.dense(net, units=1, name="values"))
            value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        with tf.variable_scope("total_loss"):
            self.loss = action_loss + value_loss * 0.5 - entropy_sum * FLAGS.entropy

        with tf.variable_scope("train_op"):
            self.optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            gradients = self.optim.compute_gradients(loss=self.loss)
            gradients = [(tf.clip_by_norm(grad, FLAGS.norm), var) for grad, var in gradients]
            self.train_op = self.optim.apply_gradients(gradients,
                                                       global_step=tf.train.get_or_create_global_step())

        tf.summary.histogram("Action Probs", self.action_probs)
        tf.summary.histogram("Entropy", entropy)
        tf.summary.histogram("Actions", self.actions)
        tf.summary.scalar("Loss/total", self.loss)
        tf.summary.scalar("Loss/actor", action_loss)
        tf.summary.scalar("Loss/value", value_loss)
        tf.summary.image("Screen", tf.gather(self.states[:, :, :, -1:], tf.random_uniform(shape=[3, ],
                                                                                          minval=0,
                                                                                          maxval=tf.shape(self.states)[0],
                                                                                          dtype=np.int32)))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("{}/main".format(FLAGS.logdir), graph=tf.get_default_graph())

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
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1)

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

    def get_actions_values(self, states):
        sess = tf.get_default_session()
        feed = {
            self.states: states,
        }

        action_probs, values = sess.run([self.action_probs, self.values], feed)
        noises = np.random.uniform(size=action_probs.shape[0])[:, np.newaxis]

        return (np.cumsum(action_probs, axis=1) > noises).argmax(axis=1), values.flatten()

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
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + FLAGS.epsilon

        advantages = rewards - values
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages) + FLAGS.epsilon

        sess = tf.get_default_session()
        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.advantages: advantages
        }
        _, summary_op, global_step = sess.run([self.train_op,
                                               self.summary_op,
                                               tf.train.get_global_step()],
                                              feed_dict=feed)
        self.summary_writer.add_summary(summary_op, global_step=global_step)


def run_episodes(envs: Iterable[gym.Env], agent: Agent, t_max=FLAGS.t_max, pipeline_fn=pipeline):
    n_envs = len(envs)
    all_dones = False

    states_memory = [[] for _ in range(n_envs)]
    actions_memory = [[] for _ in range(n_envs)]
    rewards_memory = [[] for _ in range(n_envs)]
    values_memory = [[] for _ in range(n_envs)]

    is_env_done = [False for _ in range(n_envs)]
    episode_rewards = [0 for _ in range(n_envs)]

    observations = []
    lives_info = []

    for id, env in enumerate(envs):
        env.reset()
        s, r, done, info = env.step(1)
        s = pipeline_fn(s)
        observations.append(s)

        if "Breakout" in FLAGS.env:
            lives_info.append(info['ale.lives'])

    while not all_dones:

        for t in range(t_max):

            actions, values = agent.get_actions_values(observations)

            for id, env in enumerate(envs):

                if not is_env_done[id]:

                    s2, r, is_env_done[id], info = env.step(actions[id])

                    episode_rewards[id] += r

                    if "Breakout" in FLAGS.env and info['ale.lives'] < lives_info[id]:
                        r = -1.0
                        lives_info[id] = info['ale.lives']

                    states_memory[id].append(observations[id])
                    actions_memory[id].append(actions[id])
                    rewards_memory[id].append(r)
                    values_memory[id].append(values[id])

                    observations[id] = pipeline_fn(s2)

        future_values = agent.get_values(observations)

        for id in range(n_envs):
            if not is_env_done[id] and rewards_memory[id][-1] != -1:
                rewards_memory[id][-1] += FLAGS.decay * future_values[id]

        agent.train(states_memory, actions_memory, rewards_memory, values_memory)

        states_memory = [[] for _ in range(n_envs)]
        actions_memory = [[] for _ in range(n_envs)]
        rewards_memory = [[] for _ in range(n_envs)]
        values_memory = [[] for _ in range(n_envs)]

        all_dones = np.all(is_env_done)

    return episode_rewards


def main():
    input_shape = [80, 80, 1]
    output_dim = 4
    pipeline_fn = partial(pipeline, new_HW=input_shape[:-1])

    envs = [gym.make(FLAGS.env) for i in range(FLAGS.n_envs)]
    envs[0] = gym.wrappers.Monitor(envs[0], "monitors", force=True)

    summary_writers = [tf.summary.FileWriter(logdir="{}/env-{}".format(FLAGS.logdir, i)) for i in range(FLAGS.n_envs)]
    agent = Agent(input_shape, output_dim)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)

    with tf.Session() as sess:
        try:
            if latest_checkpoint is not None:
                saver.restore(sess, latest_checkpoint)
                print("Restored from {}".format(latest_checkpoint))
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                print("Initialized weights")

            episode = 1
            while True:
                rewards = run_episodes(envs, agent, pipeline_fn=pipeline_fn)
                print(episode, np.mean(rewards))
                print(rewards)
                print()

                for id, r in enumerate(rewards):
                    summary = tf.Summary()
                    summary.value.add(tag="Episode Reward", simple_value=r)
                    summary_writers[id].add_summary(summary, global_step=episode)
                    summary_writers[id].flush()

                if episode % 10 == 0:
                    saver.save(sess, "{}/model.ckpt".format(FLAGS.logdir), write_meta_graph=False)
                    print("Saved to {}/model.ckpt".format(FLAGS.logdir))

                episode += 1

        finally:
            saver.save(sess, "{}/model.ckpt".format(FLAGS.logdir), write_meta_graph=False)
            print("Saved to {}/model.ckpt".format(FLAGS.logdir))

            for env in envs:
                env.close()

            for writer in summary_writers:
                writer.close()


if __name__ == '__main__':
    main()
