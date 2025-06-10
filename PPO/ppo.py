import argparse
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MCTS.ofcp_player import OpenFaceChinesePoker, random_bot_agent, evaluate_hand

class PPOAgent(nn.Module):
    def __init__(self, inp: int = 8, act: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(inp, 128)
        self.pi  = nn.Linear(128, act)
        self.v   = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return F.softmax(self.pi(x), dim=-1), self.v(x).squeeze(-1)


def encode_state(hand, card):
    """Return an 8-dim tensor: 6 pile-strength features + 2 card features."""
    def pile(p):
        if p:
            cat, val = evaluate_hand(p)
            val = max(val) if isinstance(val, list) else val
        else:
            cat, val = 0, 0
        return [cat, val]

    ranks, suits = '23456789TJQKA', 'CDHS'
    return torch.tensor(
        pile(hand.top) + pile(hand.middle) + pile(hand.bottom) +
        [ranks.index(card.rank), suits.index(card.suit)],
        dtype=torch.float32
    )

#  PPO-GAE training loop with additional tie penalties and scoop bonuses
def ppo_train(episodes=2000, lr=5e-4, eps=0.2, γ=0.99, λ=0.95):
    agent = PPOAgent()
    opt   = optim.Adam(agent.parameters(), lr=lr)

    # ←-- new tie/scoop parameters
    tie_penalty = -0.1     # small negative reward for a tie
    scoop_bonus = +1.0     # extra reward for winning all rows (scoop)
    max_diff    = 3        # maximum possible score-difference for a scoop

    positions = ['top', 'middle', 'bottom']
    pos2idx   = {p: i for i, p in enumerate(positions)}

    for ep in range(episodes):
        env = OpenFaceChinesePoker()

        # ───── trajectory buffers ─────
        s_list, a_idx, avail_list, old_lp = [], [], [], []
        rewards, values = [], []

        # ───── initial deal ─────
        player_cards, bot_cards = env.initial_deal()
        p_card, b_card = player_cards.pop(), bot_cards.pop()
        prev_diff = 0.0                                            # Δ(score) baseline

        # ───── play one game ─────
        while True:
            # 1) encode state and choose action
            s   = encode_state(env.player_hand, p_card)
            pi, v = agent(s)

            # legal positions
            available = [
                p for p in positions
                if len(getattr(env.player_hand, p)) < (3 if p == 'top' else 5)
            ]
            if not available:
                rewards.append(-2.0)
                values.append(v.item())
                old_lp.append(torch.tensor(0.0))
                break

            idxs = [pos2idx[p] for p in available]
            sub  = pi[idxs]
            sub  = sub / sub.sum() if sub.sum() > 0 else torch.ones_like(sub) / len(sub)
            dist = torch.distributions.Categorical(sub)
            a    = dist.sample().item()

            # record transition
            s_list.append(s)
            avail_list.append(idxs)
            a_idx.append(a)
            old_lp.append(dist.log_prob(torch.tensor(a)))
            values.append(v.item())

            # play move
            env.play_round(
                {'cards': [p_card], 'positions': [(available[a], p_card)]},
                random_bot_agent(env.bot_hand, b_card, env.player_hand)
            )

            # foul mid-game
            if not env.player_hand.valid_hand() and not env.player_hand.is_complete():
                rewards.append(-6.0)
                break

            # terminal check
            if env.game_over():
                p_sc, b_sc = env.calculate_scores()
                diff = p_sc - b_sc

                # tie penalty or scoop bonus
                if diff == 0:
                    rewards.append(tie_penalty)
                else:
                    r = diff - prev_diff
                    if diff == max_diff:
                        r += scoop_bonus
                    rewards.append(r)
                break

            # normal step reward
            p_sc, b_sc = env.calculate_scores()
            diff = p_sc - b_sc
            rewards.append(-0.1 if diff == prev_diff else diff - prev_diff)
            prev_diff = diff

            # next cards
            nxt = env.deal_next_cards()
            if not nxt or not nxt[0] or not nxt[1]:
                break
            p_card, b_card = nxt[0][0], nxt[1][0]

        # ──────────────────── GAE & returns ───────────────────
        T = len(rewards)
        values.append(0.0)                                         # bootstrap
        adv, gae = torch.zeros(T), 0.0
        for t in reversed(range(T)):
            δ   = rewards[t] + γ * values[t + 1] - values[t]
            gae = δ + γ * λ * gae
            adv[t] = gae
        returns = adv + torch.tensor(values[:-1])
        if T > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ───────────────── PPO update ─────────────────
        old_lp_t = torch.stack(old_lp)
        new_lp, ent = [], []
        for s, idxs, a in zip(s_list, avail_list, a_idx):
            pi_new, _ = agent(s)
            sub = pi_new[idxs]
            sub = sub / sub.sum()
            d = torch.distributions.Categorical(sub)
            new_lp.append(d.log_prob(torch.tensor(a)))
            ent.append(d.entropy())
        new_lp = torch.stack(new_lp)
        ent    = torch.stack(ent)

        ratio  = (new_lp - old_lp_t).exp()
        clipr  = torch.clamp(ratio, 1 - eps, 1 + eps)
        pol_loss = -(torch.min(ratio * adv, clipr * adv)).mean()
        v_loss   = F.mse_loss(torch.tensor(values[:-1]), returns)
        loss     = pol_loss + 0.5 * v_loss - 0.01 * ent.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            print(f'[EP {ep:4d}] loss={loss.item():.3f}  π-loss={pol_loss.item():.3f}  V-loss={v_loss.item():.3f}')

    return agent

def evaluate(agent, games=100):
    wins = ties = bot = tot_p = tot_b = 0
    positions = ['top', 'middle', 'bottom']
    pos2idx   = {p: i for i, p in enumerate(positions)}

    for g in range(games):
        env = OpenFaceChinesePoker()
        pc, bc = env.initial_deal()
        p_card, b_card = pc.pop(), bc.pop()

        while True:
            # encode once, then get policy distribution
            s = encode_state(env.player_hand, p_card)
            pi, _ = agent(s)

            # legal moves mask
            available = [
                p for p in positions
                if len(getattr(env.player_hand, p)) < (3 if p == 'top' else 5)
            ]
            if not available:
                break

            idxs = [pos2idx[p] for p in available]
            sub  = pi[idxs]
            sub  = sub / sub.sum() if sub.sum() > 0 else torch.ones_like(sub) / len(sub)
            dist = torch.distributions.Categorical(sub)

            # sample action
            a = dist.sample().item()

            env.play_round(
                {'cards': [p_card], 'positions': [(available[a], p_card)]},
                random_bot_agent(env.bot_hand, b_card, env.player_hand)
            )

            if env.game_over():
                break

            nxt = env.deal_next_cards()
            if not nxt or not nxt[0] or not nxt[1]:
                break
            p_card, b_card = nxt[0][0], nxt[1][0]

        p_sc, b_sc = env.calculate_scores()
        tot_p += p_sc
        tot_b += b_sc
        wins  += int(p_sc >  b_sc)
        ties  += int(p_sc == b_sc)
        bot   += int(p_sc <  b_sc)

    print('──────── Evaluation ────────')
    print(f'Wins {wins:3d} | Ties {ties:3d} | Bot wins {bot:3d}')
    print(f'Average score  Player {tot_p / games:6.2f}   Bot {tot_b / games:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',    type=int, default=2000)
    parser.add_argument('--eval_games',  type=int, default=100)
    args = parser.parse_args()

    trained = ppo_train(episodes=args.episodes)
    evaluate(trained, games=args.eval_games)
