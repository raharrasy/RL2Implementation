from Network import RL2LSTM
import torch
import torch.distributions as dist
import torch.optim as optim

class Agent(object):
    def __init__(self,input_dim, hidden_dim, output_dim, lr):
        self.LSTMNetwork = RL2LSTM(input_dim, hidden_dim, output_dim)
        self.optim = optim.Adam(self.LSTMNetwork.parameters(), lr=lr)

    def step(self, state, action, reward, done, hiddens):
        all_concat = torch.cat((state, action, reward, done), dim=-1).unsqueeze(0)
        pol_out, crit_out, hiddens = self.LSTMNetwork(all_concat, hiddens)

        return pol_out, crit_out, hiddens

    def act(self, pol):
        act_one_hots = dist.OneHotCategorical(pol)
        acts = act_one_hots.sample()
        return acts

    def update(self, batch, last_obs, last_acts, last_rews, last_dones, args, last_hiddens, timestep_left):
        rewards = [elem['rewards'] for elem in batch]
        predicted_values = [elem['predicted_values'] for elem in batch]
        log_probs = [elem['log_probs'] for elem in batch]
        entropies = [elem['entropies'] for elem in batch]

        reward_batch = torch.cat(tuple(rewards), dim=-1)
        predicted_values_batch = torch.cat(tuple(predicted_values), dim=-1)
        log_probs_batch = torch.cat(tuple(log_probs), dim=-1)
        entropies_batch = torch.cat(tuple(entropies), dim=-1)

        entropy_sums = entropies_batch.mean()
        episode_length = predicted_values_batch.size()[0]
        num_threads = predicted_values_batch.size()[1]

        R = None
        if timestep_left == 0:
            R = torch.zeros((1,num_threads))
        else:
            model_outs = [self.step(last_ob[1], last_act, last_rew, last_done,last_hidden)[1][0]
                          for last_ob, last_act, last_rew, last_done,last_hidden in
                          zip(last_obs, last_acts, last_rews, last_dones,last_hiddens)]
            R = torch.cat(tuple(model_outs), dim=-1)

        gae = torch.zeros((1,num_threads))
        sum_critic_loss, sum_policy_loss = 0, 0
        for a in reversed(range(episode_length)):
            R = reward_batch[a] + (args.gamma * R)
            adv = R.detach() - predicted_values_batch[a]
            sum_critic_loss = sum_critic_loss + 0.5 * adv.pow(2).mean()

            gae = gae * args.gamma * args.tau + adv
            sum_policy_loss = sum_policy_loss - (log_probs_batch[a] * gae.detach()).mean()
            # print("GAE :", gae)
            # print("Advantage", adv)
            # print("Rew", reward_batch[a])
            # print("log_prob : ", log_probs_batch[a])

        print("Policy loss : ",(sum_policy_loss/episode_length))
        print("Critic loss : ", args.criticLossCoef * (sum_critic_loss/episode_length))
        total_loss = -args.entropyCoef * entropy_sums +  (sum_policy_loss/episode_length) + args.criticLossCoef * (sum_critic_loss/episode_length)
        self.optim.zero_grad()
        total_loss.backward(retain_graph=True)
        if args.maxGrads is not None:
            torch.nn.utils.clip_grad_norm_(self.LSTMNetwork.parameters(), args.maxGrads)
        self.optim.step()
