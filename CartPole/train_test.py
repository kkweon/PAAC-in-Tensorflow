import train
import tensorflow as tf
import numpy as np
import gym

def vstack_test():
    n_envs = 3
    states_memory = [[] for _ in range(n_envs)]

    for i in range(n_envs):

        if random.rand() < 0.3:
            states_memory[i].append(np.random.randn(4,))


    assert 0, np.vstack(states_memory)


class AgentTest(tf.test.TestCase):

    def test_agent_can_choose_action(self):

        with self.test_session() as sess:
            agent = train.Agent(input_dim=4, output_dim=2)

            init = tf.global_variables_initializer()
            sess.run(init)
            states = np.random.randn(32, 4)
            actions = agent.get_actions(states)

            assert actions.shape == (32, )
            assert all((actions >= 0) & (actions < 2))

            states = np.random.randn(4)
            actions = agent.get_actions(states)

            assert actions.shape == (1, )
            assert all((actions >= 0) & (actions < 2))

    def test_agent_can_return_values(self):

        with self.test_session() as sess:
            agent = train.Agent(input_dim=4, output_dim=2)

            init = tf.global_variables_initializer()
            sess.run(init)
            states = np.random.randn(32, 4)
            values = agent.get_values(states)

            assert values.shape == (32, )

            states = np.random.randn(4)
            values = agent.get_values(states)

            assert values.shape == (1, )

    def test_agent_can_train(self):
        with self.test_session() as sess:
            agent = train.Agent(input_dim=4, output_dim=2)

            init = tf.global_variables_initializer()
            sess.run(init)

            n_envs = 3
            t_max = 5
            try:
                env_list = [gym.make("CartPole-v0") for i in range(n_envs)]

                state_memory = [[] for _ in range(n_envs)]
                action_memory = [[] for _ in range(n_envs)]
                reward_memory = [[] for _ in range(n_envs)]
                value_memory = [[] for _ in range(n_envs)]

                observations = [env.reset() for env in env_list]
                is_env_done = [False for env in env_list]
                total_rewards_envs = [0 for env in env_list]

                all_envs_done = False

                while not all_envs_done:

                    for t in range(t_max):

                        actions = agent.get_actions(observations)
                        values = agent.get_values(observations)

                        for idx, env in enumerate(env_list):

                            if not is_env_done[idx]:

                                state_memory[idx].append(observations[idx])
                                action_memory[idx].append(actions[idx])
                                value_memory[idx].append(values[idx])

                                observations[idx], r, is_env_done[idx], _ = env.step(actions[idx])
                                reward_memory[idx].append(r)

                                total_rewards_envs[idx] += r

                    temp_values = agent.get_values(observations)

                    for i in range(n_envs):
                        if not is_env_done[i]:
                            reward_memory[i][-1] += train.FLAGS.decay * temp_values[i]

                    agent.train(state_memory, action_memory, reward_memory, value_memory)
                    state_memory = [[] for _ in range(n_envs)]
                    action_memory = [[] for _ in range(n_envs)]
                    reward_memory = [[] for _ in range(n_envs)]
                    value_memory = [[] for _ in range(n_envs)]

                    all_envs_done = all(is_env_done)

                assert np.mean(total_rewards_envs) > 0

            finally:
                for env in env_list:
                    env.close()


def test_discounted_rewards():
    input_ = [1, 1, 1]
    gamma = .99
    expected = [1 + gamma + gamma**2, 1 + gamma, 1]

    np.testing.assert_almost_equal(expected, train.discount_rewards(input_))


def test_multi_rewards():
    input_ = [[1, 1, 1],
              [2, 1, 2],
              [3, 3, 2]]
    gamma = .99
    expected = [[1 + gamma + gamma**2, 1 + gamma, 1],
                [2 + gamma + 2 * gamma**2, 1 + 2 * gamma, 2],
                [3 + 3 * gamma + 2 * gamma**2, 3 + 2 * gamma, 2]]

    np.testing.assert_almost_equal(expected, train.discounted_multienv_rewards(input_, gamma))


def test_various_timestep_multi_rewards():

    def make_2d(n_envs):

        return [[] for _ in range(n_envs)]

    temp = make_2d(3)

    temp[0].extend([1, 2, 3])
    temp[1].extend([1, 2])
    temp[2].extend([1])

    gamma = 1

    expected = [np.array([1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3]),
                np.array([1 + 2 * gamma, 2]),
                np.array([1])]

    output = train.discounted_multienv_rewards(expected, gamma)
    assert expected == output
