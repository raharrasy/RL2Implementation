import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from torch.autograd import Variable
from env_wrappers import SubprocVecEnv, DummyVecEnv
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from ddpg import DDPG
from Network import RL2LSTM
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import torch.optim as optim
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_threads", type=int, default=16,
                    help="number_of_threads")
parser.add_argument("--seed", type=int, default=10,
                    help="seed")
parser.add_argument("--trial_length", type=int, default=2,
                    help="length of plays with same opponent")
parser.add_argument("--lstm_output_dim", type=int, default=5,
                    help="lstm hidden dimension")
parser.add_argument("--hidden_dim", type=int, default=20,
                    help="MLP hidden dim for V and Policy")
parser.add_argument("--action_space_size", type=int, default=5,
                    help="Action Space Size")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
parser.add_argument("--max_episode_length", type=int, default=25,
                    help="Maximum episode length")
parser.add_argument("--thread_update_length", type=int, default=50,
                    help="Wait length for updates")
parser.add_argument('--gamma', type=float, default=0.99,
					help='Discount Factor')
parser.add_argument('--tau', type=float, default=0.99,
					help='parameter for GAE')
parser.add_argument('--entropyCoef', type=float, default=0.001,
					help='entropy term coefficient')
parser.add_argument('--criticLossCoef', type=float, default=1.0,
					help='value loss coefficient')
parser.add_argument('--maxGrads', type=float, default=5,
					help='maximum gradient before clipped')
parser.add_argument('--eval_freq', type=float, default=200,
					help='maximum gradient before clipped')

USE_CUDA = False  # torch.cuda.is_available()
from torch.optim import Adam

def load_next_opponent():
    random_opponent_id = random.randint(1,1)
    speakers = DDPG(3, 3, 22, 64, 0.01)
    speakers.load_params(torch.load('agents/agent' + str(random_opponent_id) + '/params_25000.pt')['agent_params'][0])
    return speakers

def to_one_hot(size, index):
    output = torch.zeros((1,size))
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
    environment = make_parallel_env("simple_speaker_listener", args.num_threads, args.seed, discrete_action=True)

    ctrlr_input_size = environment.observation_space[1].shape[0] + args.action_space_size + 2
    action_space_size = args.action_space_size
    ctrlr = RL2LSTM(ctrlr_input_size, args.lstm_output_dim, args.hidden_dim,
                        action_space_size)

    optimizer = optim.Adam(ctrlr.parameters(), lr=args.lr)

    trial_number = 0
    reward_list, value_list, log_probs_list, entropy_list = [[] for idx in range(args.num_threads)], \
                                            [[] for idx in range(args.num_threads)],\
                                            [[] for idx in range(args.num_threads)],\
                                            [[] for idx in range(args.num_threads)]
    while True:
        oppo_models = [load_next_opponent() for idx in range(args.num_threads)]
        hidden_states = [(torch.zeros((1, 1, args.lstm_output_dim)), torch.zeros((1, 1, args.lstm_output_dim)))
                         for idx in range(args.num_threads)]
        num_episode = 0
        while num_episode < args.trial_length:
            raw_states = environment.reset()
            step_number = 0
            total_reward = [0 for idx in range(args.num_threads)]
            action_adds, reward_adds, done_adds = [torch.zeros((1, args.action_space_size)) for idx in range(args.num_threads)], \
                                                  [torch.zeros((1, 1)) for idx in range(args.num_threads)], \
                                                  [torch.zeros((1, 1)) for idx in range(args.num_threads)]
            dones = [False for idx in range(args.num_threads)]
            num_episode += 1
            while not any(dones):
                tensor_state = [[torch.Tensor([raw_state[0]]), torch.Tensor([raw_state[1]])] for
                                raw_state in raw_states]
                opponent_input_tensors = [thread_state[0] for thread_state in tensor_state]
                combined_input_tensors = [torch.cat((thread_state[1], action_add, reward_add, done_add), dim=-1) for
                                         thread_state, action_add, reward_add, done_add in
                                         zip(tensor_state,action_adds,reward_adds, done_adds)]
                listener_out_list = [ctrlr(combined_input_tensor, hidden_state) for
                                     combined_input_tensor, hidden_state in
                                     zip(combined_input_tensors,hidden_states)]
                speaker_out_list = [oppo_model.step(opponent_input_tensor) for oppo_model, opponent_input_tensor
                                    in zip(oppo_models,opponent_input_tensors)]
                hidden_states = [listener_out[2] for listener_out in listener_out_list]
                action_distribs = [dist.Categorical(F.softmax(listener_out[0], dim=1)) for listener_out in listener_out_list]
                action_entropy = [action_distrib.entropy() for action_distrib in action_distribs]
                listener_action_id = [action_distrib.sample().view(1, 1) for action_distrib in action_distribs]
                listener_one_hots = [to_one_hot(args.action_space_size, listener_action) for
                                     listener_action in listener_action_id]

                action_logits = [action_distrib.log_prob(torch.Tensor([[action_id]]).long())
                                 for action_distrib, action_id in zip(action_distribs, listener_action_id)]

                chosen_acts = [[speaker_out[0].numpy(), listener_one_hot[0].numpy()]
                               for speaker_out, listener_one_hot in
                               zip(speaker_out_list, listener_one_hots)]

                newObservation, rewards, dones, info = environment.step(chosen_acts)
                total_reward = [rew[0] + tot for rew, tot in zip(rewards, total_reward)]
                step_number += 1
                dones = [done[0] or (step_number == args.max_episode_length) for done in dones]

                action_adds = listener_one_hots
                reward_adds = [torch.Tensor([[reward[1]]]) for reward in rewards]
                done_adds = [torch.Tensor([[done]]) for done in dones]

                for idx in range(args.num_threads):
                    reward_list[idx].insert(0,reward_adds[idx])
                    value_list[idx].insert(0,listener_out_list[idx][1])
                    log_probs_list[idx].insert(0,action_logits[idx])
                    entropy_list[idx].insert(0,action_entropy[idx])

                update_flag = any(dones) or step_number % args.thread_update_length == 0

                if update_flag:
                    R = None

                    if any(dones):
                        R = [torch.Tensor([[0]]) for idx in range(args.num_threads)]
                    else:
                        target_state = [[torch.Tensor([raw_state[0]]), torch.Tensor([raw_state[1]])] for
                                        raw_state in newObservation]
                        combined_input_tensors = [torch.cat((thread_state[1], action_add, reward_add, done_add), dim=-1)
                                                  for thread_state, action_add, reward_add, done_add in
                                                  zip(target_state, action_adds, reward_adds, done_adds)]
                        R = [ctrlr(combined_input_tensor, hidden_state)[1].detach() for combined_input_tensor, hidden_state in
                                     zip(combined_input_tensors,hidden_states)]

                    sum_policy_loss = 0
                    sum_critic_loss = 0

                    gae = [torch.zeros(1, 1) for idx in range(args.num_threads)]

                    for seq_id in range(len(reward_list[0])):
                        for thread_id in range(args.num_threads):
                            R[thread_id] = reward_list[thread_id][seq_id] + (args.gamma * R[thread_id])
                            adv = R[thread_id] - value_list[thread_id][seq_id]
                            sum_critic_loss = sum_critic_loss + 0.5 * adv.pow(2)

                            gae[thread_id] = gae[thread_id] * args.gamma * args.tau + adv
                            entr = entropy_list[thread_id][seq_id]
                            log_prob = log_probs_list[thread_id][seq_id]
                            sum_policy_loss = sum_policy_loss - (log_prob) * gae[thread_id].detach() - args.entropyCoef * entr

                    optimizer.zero_grad()
                    loss = sum_policy_loss + args.criticLossCoef * sum_critic_loss
                    loss.backward(retain_graph = True)

                    if args.maxGrads != None:
                        torch.nn.utils.clip_grad_norm_(ctrlr.parameters(), args.maxGrads)

                    optimizer.step()

                    reward_list, value_list, log_probs_list, entropy_list = [[] for idx in range(args.num_threads)], \
                                                                            [[] for idx in range(args.num_threads)], \
                                                                            [[] for idx in range(args.num_threads)], \
                                                                            [[] for idx in range(args.num_threads)]


                if not any(dones):
                    raw_states = newObservation

            print("Total Reward episode ", str(trial_number), " : ", sum(total_reward)/args.num_threads)
        trial_number += 1
        if trial_number % 100 == 0 :
            eval(args, ctrlr)



def eval(args, model):
    environment = make_env("simple_speaker_listener", discrete_action=True)
    ctrlr = model

    episode_number = 0
    while episode_number < 10:
        oppo_model = load_next_opponent()
        hidden_states = (torch.zeros((1, 1, args.lstm_output_dim)), torch.zeros((1, 1, args.lstm_output_dim)))
        total_reward = 0
        for a in range(args.trial_length):
            raw_states = environment.reset()
            step_number = 0
            tensor_state = [torch.Tensor([raw_states[0]]), torch.Tensor([raw_states[1]])]
            action_adds = torch.zeros((1, args.action_space_size))
            reward_adds = torch.zeros((1, 1))
            environment.render()
            done_adds = torch.zeros((1, 1))
            opponent_input_tensor = tensor_state[0]
            combined_input_tensor = torch.cat((tensor_state[1], action_adds, reward_adds, done_adds), dim=-1)
            done = False
            while not done:
                tensor_state = [torch.Tensor([raw_states[0]]), torch.Tensor([raw_states[1]])]
                opponent_input_tensor = tensor_state[0]
                combined_input_tensor = torch.cat((tensor_state[1], action_adds, reward_adds, done_adds), dim=-1)
                listener_out_list = ctrlr(combined_input_tensor, hidden_states)
                speaker_out_list = oppo_model.step(opponent_input_tensor)

                hidden_states = listener_out_list[2]
                print(F.softmax(listener_out_list[0],dim=1))
                action_distribs = dist.Categorical(F.softmax(listener_out_list[0],dim=1))
                listener_action_id = action_distribs.sample().view(1,1)
                listener_one_hots = to_one_hot(args.action_space_size, listener_action_id)

                chosen_acts = [speaker_out_list[0].numpy(), listener_one_hots[0].numpy()]
                newObservation, reward, done, info = environment.step(chosen_acts)
                environment.render()
                total_reward += reward[0]
                step_number += 1
                done = done[0] or (step_number==args.max_episode_length)

                action_adds = listener_one_hots
                reward_adds = torch.Tensor([[reward[1]]])
                done_adds = torch.Tensor([[done]])

                if not done:
                    raw_states = newObservation
        episode_number += 1

        print("Total Reward episode ", str(episode_number), " : ", total_reward)

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)