import train
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_image(image, **kwargs):
    image = np.squeeze(image)
    shape = image.shape

    if len(shape) == 2:
        plt.imshow(image, cmap="gray", **kwargs)
    else:
        plt.imshow(image, **kwargs)


def test_gym_can_make_breakout():
    env = gym.make("Breakout-v0")

    assert env.action_space.n == 4
    assert env.observation_space.shape == (210, 160, 3)


def run_pipeline():
    env = gym.make("Breakout-v0")
    s = env.reset()
    env.step(1)

    for _ in range(100):
        # 0 - nothing
        # 1 - fire
        # 2 - right
        # 3 - left
        s, r, done, info = env.step(env.action_space.sample())
        # info = {'ale.lives': 5}

        plot_image(train.pipeline(s, new_HW=(80, 80)))
        plt.title("R: {}, Done: {}, Info: {}".format(r, done, info))
        plt.show()
        if r != 0:
            print(r, done, info)

    env.close()


def test_vstack():
    states = [[np.random.randn(32, 32, 1),
               np.random.randn(32, 32, 1),
               np.random.randn(32, 32, 1)],
              [],
              [np.random.randn(32, 32, 1)]]

    assert np.vstack([item for item in states if len(item) > 0]).shape == (4, 32, 32, 1)


def test_discount_rewards():
    rewards = [1]
    gamma = .99
    expect = [1]
    np.testing.assert_almost_equal(expect, train.discount_rewards(rewards, gamma))

    rewards = [-1, 0, 0, -1, 1]
    gamma = .99

    expect = [-1,
              -1 * gamma ** 2,
              gamma * -1,
              -1,
              1]
    output = train.discount_rewards(rewards, gamma)

    np.testing.assert_almost_equal(expect, output)

    rewards = [-1, -2, -3]
    gamma = .99
    expect = [-1, -2, -3]
    np.testing.assert_almost_equal(expect, train.discount_rewards(rewards, gamma))

    rewards = [1, 2, 3, 4]
    gamma = 1
    expect = [1 + 2 * gamma + 3 * gamma**2 + 4 * gamma**3,
              2 + 3 * gamma + 4 * gamma**2,
              3 + 4 * gamma,
              4]
    np.testing.assert_almost_equal(expect, train.discount_rewards(rewards, gamma))


def test_multi_discount_rewards():
    gamma = .99
    r1 = [-1, 0, 0, -1, 1]
    e1 = [-1, -1 * gamma ** 2, gamma * -1, -1, 1]

    r2 = [-1, -2, -3]
    e2 = [-1, -2, -3]

    r3 = [1, 2, 3, 4]
    e3 = [1 + 2 * gamma + 3 * gamma**2 + 4 * gamma**3,
          2 + 3 * gamma + 4 * gamma**2,
          3 + 4 * gamma,
          4]

    rewards = [r1, r2, r3]
    expect = [e1, e2, e3]
    expect = [np.array(i, dtype=np.float32) for i in expect]

    out = train.discount_multi_rewards(rewards, gamma)

    assert len(out) == len(expect)

    for left, right in zip(expect, out):
        np.testing.assert_almost_equal(left, right)

    assert np.hstack(out).shape == (12,)
    np.testing.assert_almost_equal(np.hstack(out), np.hstack(expect))


def test_agent():
    input_shape = [40, 40, 1]
    output_dim = 4
    agent = train.Agent(input_shape, output_dim)
    assert hasattr(agent, "states")
    assert hasattr(agent, "actions")
    assert hasattr(agent, "rewards")
    assert hasattr(agent, "advantages")

    assert hasattr(agent, "action_probs")
    assert hasattr(agent, "values")
    assert hasattr(agent, "loss")
    assert hasattr(agent, "train_op")


class AgentTest(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.env = gym.make("Breakout-v0")
        self.g = tf.Graph()
        self.input_shape = [40, 40, 1]
        self.output_dim = self.env.action_space.n

        with self.g.as_default():
            self.agent = train.Agent(self.input_shape, self.output_dim)

    def tearDown(self):

        self.env.close()

    def test_agent_can_run_train_op(self):

        with self.test_session(graph=self.g) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            s = np.random.randn(32, *self.input_shape)
            a = np.random.randint(low=0, high=self.output_dim - 1, size=32)
            advantage = r = np.random.randn(32)

            feed = {
                self.agent.states: s,
                self.agent.actions: a,
                self.agent.rewards: r,
                self.agent.advantages: advantage
            }

            sess.run(self.agent.train_op, feed)

    def test_agent_can_get_actions(self):
        with self.test_session(graph=self.g) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            s = self.env.reset()
            s = train.pipeline(s, self.input_shape[:-1])

            actions = self.agent.get_actions(s)
            assert np.shape(actions) == (1,)
            assert np.all((actions >= 0) & (actions < self.output_dim))

            s = np.random.randn(32, *self.input_shape)
            actions = self.agent.get_actions(s)

            assert np.shape(actions) == (32,)
            assert np.all((actions >= 0) & (actions < self.output_dim))

    def test_agent_can_get_values(self):
        with self.test_session(graph=self.g) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            s = self.env.reset()
            s = train.pipeline(s, self.input_shape[:-1])

            values = self.agent.get_values(s)
            assert np.shape(values) == (1,)

            s = [s for _ in range(10)]
            s = np.array(s)

            assert s.shape == (10, *self.input_shape)
            values = self.agent.get_values(s)
            assert np.shape(values) == (10,)

    def test_agent_can_train(self):
        n_envs = 3
        envs_list = [gym.make("Breakout-v0") for _ in range(n_envs)]

        with self.test_session(graph=self.g) as sess:

            init = tf.global_variables_initializer()
            sess.run(init)

            states_memory = [[] for _ in range(n_envs)]
            actions_memory = [[] for _ in range(n_envs)]
            rewards_memory = [[] for _ in range(n_envs)]
            values_memory = [[] for _ in range(n_envs)]

            observations = [train.pipeline(env.reset(), self.input_shape[:-1]) for env in envs_list]
            dones = [False for _ in range(n_envs)]

            actions = self.agent.get_actions(observations)
            values = self.agent.get_values(observations)

            for id, env in enumerate(envs_list):

                s2, r, dones[id], _ = env.step(actions[id])

                states_memory[id].append(observations[id])
                actions_memory[id].append(actions[id])
                rewards_memory[id].append(r)
                values_memory[id].append(values[id])

                observations[id] = train.pipeline(s2, self.input_shape[:-1])
            self.agent.train(states_memory, actions_memory, rewards_memory, values_memory)

    def test_can_run_full_episodes(self):
        n_envs = 3
        try:
            agent = train.Agent(input_shape=[80, 80, 4], output_dim=4)
            envs = [gym.make("Breakout-v0") for _ in range(n_envs)]
            pipeline_fn = lambda x: train.pipeline(x, [80, 80])

            with self.test_session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                print(train.run_episodes(envs, agent, pipeline_fn=pipeline_fn))

        finally:
            for env in envs:
                env.close()


if __name__ == '__main__':
    run_pipeline()
