import torch
import numpy as np
from env_wrappers import SubprocVecEnv, DummyVecEnv
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from ddpg import DDPG
from Network import RL2LSTM
import torch.distributions as dist
import torch.nn.functional as F
import torch.optim as optim
import random
from misc import onehot_from_logits
from agent import Agent
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_threads", type=int, default=10,
                    help="number_of_threads")
parser.add_argument("--seed", type=int, default=10,
                    help="seed")
parser.add_argument("--trial_length", type=int, default=2,
                    help="length of plays with same opponent")
parser.add_argument("--lstm_output_dim", type=int, default=5,
                    help="lstm hidden dimension")
parser.add_argument("--hidden_dim", type=int, default=64,
                    help="MLP hidden dim for V and Policy")
parser.add_argument("--action_space_size", type=int, default=5,
                    help="Action Space Size")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="Learning rate")
parser.add_argument("--max_episode_length", type=int, default=25,
                    help="Maximum episode length")
parser.add_argument("--thread_update_length", type=int, default=5,
                    help="Wait length for updates")
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount Factor')
parser.add_argument('--tau', type=float, default=0.99,
                    help='parameter for GAE')
parser.add_argument('--entropyCoef', type=float, default=1e-4,
                    help='entropy term coefficient')
parser.add_argument('--criticLossCoef', type=float, default=5.0,
                    help='value loss coefficient')
parser.add_argument('--maxGrads', type=float, default=1,
                    help='maximum gradient before clipped')
parser.add_argument('--eval_freq', type=float, default=10,
                    help='maximum gradient before clipped')
parser.add_argument('--truncate_grad', type=bool, default=False,
                    help='maximum gradient before clipped')

USE_CUDA = False  # torch.cuda.is_available()


def load_next_opponent():
    random_opponent_id = random.randint(1, 5)
    speakers = DDPG(3, 3, 22, 64, 0.01)
    speakers.load_params(torch.load('agents/agent' + str(random_opponent_id) + '/params_25000.pt')['agent_params'][0])
    return speakers


def to_one_hot(size, index):
    output = torch.zeros((1, size))
    output[0][index.item()] = 1
    return output


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env

        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_env(scenario_name, benchmark=False, discrete_action=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation,
                            discrete_action=discrete_action)
    return env



def train(args):
    environment = [make_env("simple_speaker_listener", discrete_action=True) for a in range(args.num_threads)]
    for rank in range(len(environment)):
        environment[rank].seed(rank*1000)
    obs_dim, act_dim = 11, 5
    agent = Agent(obs_dim + act_dim + 2, args.hidden_dim, act_dim, args.lr)
    episode_number = 0
    concat_elements = [{'timestep_left': 0} for idx in range(args.num_threads)]
    hidden_elements = [None for idx in range(args.num_threads)]

    eval_results = []

    while episode_number < 8000:
        for idx in range(args.num_threads):
            a = concat_elements[idx]
            if a['timestep_left'] == 0:
                concat_elements[idx]['timestep_left'] = args.max_episode_length
                concat_elements[idx]['state'] = [torch.Tensor([obs]) for obs in environment[idx].reset()]
                concat_elements[idx]['actions'] = torch.zeros((1, 5))
                concat_elements[idx]['rewards'] = torch.zeros((1, 1))
                concat_elements[idx]['dones'] = torch.ones((1, 1))

                if episode_number % args.trial_length == 0:
                    opponent_models = [load_next_opponent() for idx in range(args.num_threads)]
                    hidden_elements[idx] = (torch.zeros(1, 1, args.hidden_dim), torch.zeros(1, 1, args.hidden_dim))

        list_of_losses = []
        last_states = []
        last_actions = []
        last_rewards = []
        last_dones = []
        last_hiddens = []
        for idx in range(args.num_threads):
            loss_elements, hidden_states = get_rollouts(environment[idx], agent, args, hidden_elements[idx],
                         concat_elements[idx], opponent_models[idx], args.thread_update_length)
            hidden_elements[idx] = hidden_states
            list_of_losses.append(loss_elements)
            last_states.append(concat_elements[idx]['state'])
            last_actions.append(concat_elements[idx]['actions'])
            last_rewards.append(concat_elements[idx]['rewards'])
            last_dones.append(concat_elements[idx]['dones'])
            last_hiddens.append(hidden_states)

        agent.update(list_of_losses, last_states, last_actions, last_rewards,last_dones,
                     args, last_hiddens, concat_elements[0]['timestep_left'])
        episode_number += 1
        if episode_number % args.eval_freq == 0:
            returns = eval(args, agent)
            print("Done ",(episode_number/args.eval_freq))
            average_return = sum(returns)/len(returns)
            eval_results.append(average_return)
            print("Eval Results : ", eval_results)
            if len(eval_results) > 10:
                f = plt.figure()
                plt.plot(eval_results)
                plt.ylim([-40.0, 0.0])
                f.savefig("results.pdf", bbox_inches='tight')




def get_rollouts(environment, agent, args, hidden_states, concat_elements, opponent, rollout_length=10, timesteps_left=25):
    states = concat_elements['state']
    prev_actions = concat_elements['actions']
    prev_rewards = concat_elements['rewards']
    prev_dones = concat_elements['dones']
    timesteps_left = concat_elements['timestep_left']

    loss_elements = {}
    loss_elements['rewards'] = torch.zeros((rollout_length, 1))
    loss_elements['predicted_values'] = torch.zeros((rollout_length, 1))
    loss_elements['log_probs'] = torch.zeros((rollout_length, 1))
    loss_elements['entropies'] = torch.zeros((rollout_length, 1))

    for a in range(min(rollout_length,timesteps_left)):
        policy, value, hidden_states = agent.step(states[1], prev_actions, prev_rewards, prev_dones, hidden_states)
        loss_elements['predicted_values'][a] = value
        agent_acts = agent.act(policy)
        opponent_acts = opponent.step(states[0], explore=False)
        new_observations, rewards, dones, infos = environment.step([opponent_acts.numpy()[0], agent_acts.numpy()[0][0]])
        dones = [(timesteps_left == min(rollout_length, timesteps_left) and a == (timesteps_left-1)) for idx in range(2)]
        loss_elements['rewards'][a] = torch.Tensor([[rewards[0]]])
        log_policies = (torch.log(policy+1e-20) * agent_acts).sum(dim=-1)
        loss_elements['log_probs'][a] = log_policies
        entropy = (-torch.log(policy+1e-20) * policy).sum(dim=-1)
        loss_elements['entropies'][a] = entropy

        states = [torch.Tensor([n_obs]) for n_obs in new_observations]
        prev_actions = agent_acts[0]
        prev_rewards = torch.Tensor([[rewards[0]]])
        prev_dones = torch.Tensor([[dones[0]]])
    if loss_elements['rewards'].std().item() > 1e-6 :
        loss_elements['rewards'] = (loss_elements['rewards'] - loss_elements['rewards'].mean()) / (loss_elements['rewards'].std())
    else :
        loss_elements['rewards'] = torch.zeros_like(loss_elements['rewards'])
    concat_elements['state'] = states
    concat_elements['actions'] = prev_actions
    concat_elements['rewards'] = prev_rewards
    concat_elements['dones'] = prev_dones
    concat_elements['timestep_left'] = timesteps_left - min(timesteps_left, rollout_length)

    return loss_elements, hidden_states


def eval(args, agent):
    environment = make_env("simple_speaker_listener", discrete_action=True)

    episode_number = 0
    returns_obtained = []
    while episode_number < 50:
        oppo_model = load_next_opponent()
        hidden_states = (torch.zeros((1, 1, args.hidden_dim)), torch.zeros((1, 1, args.hidden_dim)))
        total_reward = 0
        for a in range(args.trial_length):
            raw_states = environment.reset()
            episode_reward = 0
            step_number = 0
            tensor_state = [torch.Tensor([raw_states[0]]), torch.Tensor([raw_states[1]])]
            action_adds = torch.zeros((1, args.action_space_size))
            reward_adds = torch.zeros((1, 1))
            #environment.render()
            done_adds = torch.zeros((1, 1))
            opponent_input_tensor = tensor_state[0]
            combined_input_tensor = torch.cat((tensor_state[1], action_adds, reward_adds, done_adds), dim=-1)
            done = False
            while not done:
                tensor_state = [torch.Tensor([raw_states[0]]), torch.Tensor([raw_states[1]])]
                opponent_input_tensor = tensor_state[0]
                #combined_input_tensor = torch.cat((tensor_state[1], action_adds, reward_adds, done_adds), dim=-1)
                #listener_out_list = ctrlr(combined_input_tensor, hidden_states)

                policy, _, hidden_states = agent.step(tensor_state[1], action_adds, reward_adds,done_adds, hidden_states)
                listener_act = onehot_from_logits(policy)
                #listener_act = agent.act(policy)
                speaker_out_list = oppo_model.step(opponent_input_tensor, explore=False)

                print(F.softmax(policy, dim=-1))
                #action_distribs = dist.Categorical(F.softmax(listener_out_list[0], dim=1))
                #listener_action_id = action_distribs.sample().view(1, 1)
                #listener_one_hots = to_one_hot(args.action_space_size, listener_action_id)

                chosen_acts = [speaker_out_list[0].numpy(), listener_act[0][0].numpy()]
                newObservation, reward, done, info = environment.step(chosen_acts)
                #environment.render()
                total_reward += reward[0]
                episode_reward += reward[0]
                step_number += 1
                done = done[0] or (step_number == args.max_episode_length)

                action_adds = listener_act[0]
                reward_adds = torch.Tensor([[reward[1]]])
                done_adds = torch.Tensor([[done]])

                if not done:
                    raw_states = newObservation
            f = open("output.txt", "a")
            f.write(str(episode_reward) + "\n")
            returns_obtained.append(episode_reward)
            f.close()
        episode_number += 1
    return returns_obtained


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
