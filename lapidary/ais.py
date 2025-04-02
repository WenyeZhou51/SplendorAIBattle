from lapidary.data import colours
from lapidary.game import GameState
from lapidary.aibase import AI, MoveInfo

import matplotlib.pyplot as plt

import numpy as np

import argparse
import sys
import time
from glob import glob
from os.path import join

from lapidary.nn import H50AI, H50AI_TDlam
import random

from collections import namedtuple

# Import PPO Bot
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(current_dir), 'Tims_splendor_agent'))
try:
    from ppo_bot import PPOBotAI
except ImportError:
    pass  # Gracefully handle if PPO bot is not available

ProgressInfo = namedtuple(
    'ProgressInfo',
    ['first_player_winrate',
     'game_length',
     'winner_num_bought',
     'winner_t1_bought',
     'winner_t2_bought',
     'winner_t3_bought',
     'probabilities',
     'game_length_std',
     'sample_game_lengths',
     'g1_evaluations',
     'g2_evaluations',
     'g3_evaluations',
     'g1_state_vectors',
     'g1_scores',
     'g2_scores',
     'g3_scores',
     'g1_cards_played',
     'g2_cards_played',
     'g3_cards_played',
     'g1_supply_gems',
     'g2_supply_gems',
     'g3_supply_gems',
     'weight_sample'])

class RandomAI(AI):
    '''Chooses random moves, with preference given to buying, then
    reserving, then gems.

    '''

    def make_move(self, state):
        moves = state.get_valid_moves(state.current_player_index)
        # return state.generator.choice(moves)

        if state.players[state.current_player_index].total_num_gems <= 8:
            gems_moves = [move for move in moves if move[0] == 'gems']
            if gems_moves and state.total_num_gems_available() >= 3:
                choice = gems_moves[state.generator.randint(len(gems_moves))]
                return choice, MoveInfo(choice)
                

        buying_moves = [move for move in moves if move[0] == 'buy_available' or move[0] == 'buy_reserved']
        if buying_moves:
            choice = buying_moves[state.generator.randint(len(buying_moves))]
            return choice, MoveInfo(choice)

        reserving_moves = [move for move in moves if move[0] == 'reserve']
        if reserving_moves:
            choice = reserving_moves[state.generator.randint(len(reserving_moves))]
            return choice, MoveInfo(choice)

        choice = moves[state.generator.randint(len(moves))]
        return choice, MoveInfo(choice)
      
    def choose_noble(self, state, nobles):
        return state.generator.choice(nobles)


class FinishedGameInfo(object):
    def __init__(self, length, winner_index,
                 winner_num_t1_bought=0,
                 winner_num_t2_bought=0,
                 winner_num_t3_bought=0,
                 state_vectors=[]):
        self.length = length
        self.winner_index = winner_index
        self.winner_num_t1_bought = winner_num_t1_bought
        self.winner_num_t2_bought = winner_num_t2_bought
        self.winner_num_t3_bought = winner_num_t3_bought

        self.state_vectors = state_vectors

    @property
    def winner_num_bought(self):
        return self.winner_num_t1_bought + self.winner_num_t2_bought + self.winner_num_t3_bought

    def full_game_values(self, player_index):
        return np.vstack([gi.post_move_values[player_index] for gi in self.state_vectors])

    def full_game_scores(self, player_index):
        return np.array([gi.post_move_scores[player_index] for gi in self.state_vectors])

    def full_game_cards_played(self, player_index):
        return np.vstack([gi.post_move_cards_each_tier[player_index] for gi in self.state_vectors])

    def full_game_num_gems_in_supply(self):
        return np.array([gi.post_move_num_gems_in_supply for gi in self.state_vectors])



class GameManager(object):
    def __init__(self, players=2, ais=[], end_score=15, validate=True):
        self.end_score = end_score
        self.num_players = players

        self.validate = validate

        assert len(ais) == players

        self.ais = ais

    def run_game(self, verbose=True):
        # print('===')
        state = GameState(players=self.num_players, init_game=True,
                          validate=self.validate)
        self.state = state

        state_vectors = []
        # Add the judgement of the first state vector
        ai = self.ais[0]
        if isinstance(ai, H50AI_TDlam):
            state_vector = state.get_state_vector(0).reshape((1, -1))
            current_value = ai.session.run(ai.softmax_output, {ai.input_state: state_vector})[0]
        else:
            state_vector = np.zeros(5)
            current_value = np.zeros(2)
        # current_grads = ai.session.run(ai.grads, feed_dict={ai.input_state: state_vector})
        # state_vectors.append((state.get_state_vector(0), current_value, state_vector, current_grads))

        game_round = 1
        state = state
        while True:
            if state.current_player_index == 0:
                scores = np.array([player.score for player in state.players])
                if np.any(scores >= self.end_score):  # game has ended
                    max_score = np.max(scores)
                    num_cards = []
                    for i, player in enumerate(state.players):
                        if player.score == max_score:
                            num_cards.append(len(player.cards_played))
                    min_num_cards = np.min(num_cards)

                    winning_players = []
                    for i, player in enumerate(state.players):
                        if player.score == max_score and len(player.cards_played) == min_num_cards:
                            winning_players.append((i, player))

                    assert len(winning_players) >= 1
                    
                    print('## players {} win with {} points after {} rounds'.format([p[0] for p in winning_players], max_score, game_round))

                    try:
                        state.verify_state()
                    except AssertionError:
                        import traceback
                        traceback.print_exc()
                        import ipdb
                        ipdb.set_trace()

                    if len(winning_players) > 1:
                        print('Draw!')
                        return FinishedGameInfo(None, None, state_vectors=state_vectors)

                    winner_index = winning_players[0][0]

                    winner_num_bought = len(state.players[winner_index].cards_played)

                    winner_value = np.zeros(self.num_players)
                    winner_value[winner_index] = 1.
                    for player_index in range(state.num_players):
                        state_vectors[-1].post_move_values[player_index] = np.roll(winner_value, -1 * player_index)

                    # assert ((winner_index + 1) % state.num_players) == state.current_player_index

                    winner_t1 = len([c for c in state.players[winner_index].cards_played if c.tier == 1])
                    winner_t2 = len([c for c in state.players[winner_index].cards_played if c.tier == 2])
                    winner_t3 = len([c for c in state.players[winner_index].cards_played if c.tier == 3])
                    return FinishedGameInfo(game_round, winner_index,
                                            winner_num_t1_bought=winner_t1,
                                            winner_num_t2_bought=winner_t2,
                                            winner_num_t3_bought=winner_t3,
                                            state_vectors=state_vectors)
                    # return game_round, i, winner_num_bought, state_vectors

            if state.current_player_index == 0:
                game_round += 1
                if verbose:
                    print('Round {}: {}'.format(game_round, state.get_scores()))
            if game_round > 50:
                print('Stopped after 50 rounds')
                return FinishedGameInfo(None, None, state_vectors=state_vectors)
                # return game_round, None, None, state_vectors
            # scores = state.get_scores()
            # if any([score >= self.end_score for score in scores]):
            #     break

            # game end

            for tier in range(1, 4):
                state.generator.shuffle(state.cards_in_deck(tier))

            current_player_index = state.current_player_index
            
            move, move_info = self.ais[state.current_player_index].make_move(state)
            # print(state.current_player_index, move, values[:-1])
            if verbose:
                print('P{}: {}, value {}'.format(state.current_player_index, move, values))
            last_player = state.current_player_index
            state.make_move(move)

            # new_state_vector = state.get_state_vector(current_player_index)
            state_vectors.append(move_info)

        # winner_index = np.argmax(scores)
        # if verbose:
        #     state.print_state()
        # print('Ended with scores {} after {} rounds'.format(scores, game_round))

        # winner_num_bought = len(state.players[winner_index].cards_played)

        # return game_round, winner_index, winner_num_bought, state_vectors

ais = {'nn2ph50': H50AI(),
       # 'nn2ph50_td': H50AI_TD(),
       'random': H50AI()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=2)
    parser.add_argument('--end-score', type=int, default=15)
    parser.add_argument('--number', type=int, default=1)
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--stepsize', type=float, default=0.05)
    parser.add_argument('--debug-after-test', action='store_true', default=False)
    parser.add_argument('--train-steps', type=int, default=500)
    parser.add_argument('--prob-factor', type=float, default=10)
    parser.add_argument('--learning-half-life', type=float, default=0.)
    parser.add_argument('--no-train', default=False, action='store_true')

    parser.add_argument('--no-validate', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])

    ai = H50AI_TDlam(restore=args.restore, stepsize=args.stepsize, prob_factor=args.prob_factor, num_players=args.players)
    ais = [ai for _ in range(args.players)]
    # ais = [ai, RandomAI()]
    # ais = [RandomAI(), RandomAI()]
    # ais = [)] + [RandomAI() for _ in range(args.players - 1)]
    manager = GameManager(players=args.players, ais=ais,
                          end_score=args.end_score,
                          validate=(not args.no_validate))

    if len(glob(join('saves', '{}.*'.format(ai.name)))) and not args.restore:
        print('restore information present but not asked to use')
        exit(1)

    import tensorflow as tf
    # ai.session.run(tf.global_variables_initializer())
    # ai.load_variables()

    test_state = GameState(players=args.players, init_game=True)
    test_moves = test_state.get_valid_moves(0)
    test_state_vector = test_state.get_state_vector(0)

    stepsize_multiplier = 1.
    learning_rate_half_life = args.learning_half_life * args.train_steps
    learning_rate_halved_at = []

    round_nums = []
    t_before = time.time()
    round_collection = []
    training_data = []
    progress_info = []
    cur_time = time.time()
    try:
        for i in range(args.number):

            if i % 100 == 0 and i > 50:
                print('Saving')
                ai.saver.save(ai.session, ai.ckpt_filen())

            if args.learning_half_life > 0. and i % learning_rate_half_life == 0 and i > 2:
                print('Halving stepsize multiplier')
                stepsize_multiplier /= 2.
                learning_rate_halved_at.append(i / args.train_steps)

            # num_rounds, winner_index, winner_num_bought, state_vectors = manager.run_game(verbose=False)
            # manager.end_score = random.randint(1, 4)
            # manager.end_score = 1
            # manager.end_score = 1
            game_info = manager.run_game(verbose=False)
            if game_info.winner_index is not None:
                training_data.append((game_info.winner_index, game_info.state_vectors))
                round_collection.append(game_info)
            # if game_info.winner_index is None:
                # print('Stopped with winner None')
                # continue
            round_nums.append(game_info.length)

            # train the ai
            if i % args.train_steps == 0 and i > 2 and not args.no_train:

                print('training...', len(training_data))
                ai.train(training_data,
                         stepsize_multiplier=stepsize_multiplier,
                         stepsize=args.stepsize)
                training_data = []
                print('done')


            if i % args.train_steps == 0: # This happens even on the first run
                new_time = time.time()
                print('======== round {} / {}'.format(i, args.number))
                ai.print_info()
                new_states = [test_state.copy().make_move(move) for move in test_moves]
                vectors = np.array([new_state.get_state_vector(0) for new_state in new_states])
                outputs = ai.session.run(ai.output, {ai.input_state: vectors})
                softmax_outputs = ai.session.run(ai.softmax_output, {ai.input_state: vectors})
                probabilities = ai.session.run(ai.probabilities, {ai.input_state: vectors})

                weight_1 = ai.session.run(ai.weight_1)
                bias_1 = ai.session.run(ai.bias_1)
                weight_2 = ai.session.run(ai.weight_2)
                bias_2 = ai.session.run(ai.bias_2)
                print('test outputs:')
                for row in zip(outputs, softmax_outputs):
                    print(row[0], row[1])
                print('test probabilities:')
                for move, prob in zip(test_moves, probabilities[0]):
                    print('{:.05f}% : {}'.format(prob * 100, move))
                if args.debug_after_test:
                    import ipdb
                    ipdb.set_trace()
                # round_collection = np.array(round_collection)
                if len(round_collection) > 1:
                    progress_info.append(ProgressInfo(
                        np.average([gi.winner_index == 0 for gi in round_collection]),
                        np.average([gi.length for gi in round_collection]),
                        np.average([gi.winner_num_bought for gi in round_collection]),
                        np.average([gi.winner_num_t1_bought for gi in round_collection]),
                        np.average([gi.winner_num_t2_bought for gi in round_collection]),
                        np.average([gi.winner_num_t3_bought for gi in round_collection]),
                        probabilities[0],
                        np.std([gi.length for gi in round_collection]),
                        [gi.length for gi in round_collection],
                        (round_collection[-1].full_game_values(0), round_collection[-1].full_game_values(1)),
                        (round_collection[-2].full_game_values(0), round_collection[-2].full_game_values(1)),
                        (round_collection[-3].full_game_values(0), round_collection[-3].full_game_values(1)),
                        round_collection[-1].state_vectors,
                        (round_collection[-1].full_game_scores(0), round_collection[-1].full_game_scores(1)),
                        (round_collection[-2].full_game_scores(0), round_collection[-2].full_game_scores(1)),
                        (round_collection[-3].full_game_scores(0), round_collection[-3].full_game_scores(1)),
                        (round_collection[-1].full_game_cards_played(0), round_collection[-1].full_game_cards_played(1)),
                        (round_collection[-2].full_game_cards_played(0), round_collection[-2].full_game_cards_played(1)),
                        (round_collection[-3].full_game_cards_played(0), round_collection[-3].full_game_cards_played(1)),
                        round_collection[-1].full_game_num_gems_in_supply(),
                        round_collection[-2].full_game_num_gems_in_supply(),
                        round_collection[-3].full_game_num_gems_in_supply(),
                        weight_1[:, -2:]))

                    print('in last {} rounds, player 1 won {:.02f}%, average length {} rounds'.format(
                        args.train_steps,
                        progress_info[-1][0] * 100,
                        progress_info[-1][1]))

                round_collection = []

                print('Time per game: {}'.format((new_time - cur_time) / args.train_steps))
                cur_time = time.time()
                print('Current learning rate multiplier: {}'.format(stepsize_multiplier))

            if (i % args.train_steps == 0) and (i > 2):
                print('plotting')
                fig, axes = plt.subplots(ncols=6, nrows=3)
                ax1, ax2, ax3, ax7, ax8, ax9 = axes[0]
                ax4, ax5, ax6, ax10, ax11, ax12 = axes[1]
                ax16, ax17, ax18, ax13, ax14, ax15 = axes[2]
                # ax7, ax8, ax9 = axes[2]

                ys0 = [i[0] for i in progress_info]
                ax1.plot(np.arange(len(progress_info)) * args.train_steps,
                         [i[0] for i in progress_info])
                if len(ys0) > 4:
                    av = rolling_average(ys0)
                    av = np.hstack([[av[0]], [av[0]], av])
                    ax1.plot(np.arange(len(progress_info))[:-2] * args.train_steps, av)
                ax1.set_xlabel('num games')
                ax1.set_ylabel('player 1 winrate')
                ax1.set_ylim(0, 1)
                ax1.grid()

                ys1 = np.array([i[1] for i in progress_info])
                ax2.plot(np.arange(len(progress_info)) * args.train_steps,
                         [i[1] for i in progress_info])
                if len(ys1) > 4:
                    av = rolling_average(ys1)
                    av = np.hstack([[av[0]], [av[0]], av])
                    ax2.plot(np.arange(len(progress_info))[:-2] * args.train_steps, av)
                ax2.set_xlabel('num games')
                ax2.set_ylabel('average length')
                ax2.grid()

                ax3.plot(np.arange(len(progress_info)) * args.train_steps,
                         [i[6] for i in progress_info])
                ax3.set_xlabel('num games')
                ax3.set_ylabel('probabilities')
                ax3.set_yscale('log')
                ax3.set_ylim(10**-3, 10**0)

                ys1 = [i[2] for i in progress_info]
                ax4.plot(np.arange(len(progress_info)) * args.train_steps,
                         [i[2] for i in progress_info])
                if len(ys1) > 4:
                    av = rolling_average(ys1)
                    av = np.hstack([[av[0]], [av[0]], av])
                    ax4.plot(np.arange(len(progress_info))[:-2] * args.train_steps, av)
                ax4.set_xlabel('num games')
                ax4.set_ylabel('average winner cards played')
                ax4.grid()

                ys1 = [i[3] for i in progress_info]
                xs = np.arange(len(progress_info)) * args.train_steps
                ax5.plot(xs, [i[3] for i in progress_info], label='tier 1', color='C2')
                ax5.plot(xs, [i[4] for i in progress_info], label='tier 2', color='C3')
                ax5.plot(xs, [i[5] for i in progress_info], label='tier 3', color='C4')
                ax5.set_xlabel('num games')
                ax5.set_ylabel('average winner cards played each tier')
                ax5.grid()
                ax5.legend()
                
                num = min(10, len(progress_info) - 1)
                for i in range(num):
                    data = progress_info[-num + i][8]
                    ax6.hist(data, alpha=(i + 1) / 11., normed=True, label='dataset {}'.format(-num + i),
                             bins=np.arange(22.5, 50.5, 1))
                all_data = np.hstack([row[8] for row in progress_info[-10:]])
                ax6.hist(all_data, color='red', histtype='step', normed=True,
                         label='all', linewidth=2, bins=np.arange(22.5, 50.5, 1))
                ax6.legend(fontsize=8)
                ax6.set_xticks(np.arange(25, 51, 5))
                ax6.set_xlabel('length')
                ax6.set_ylabel('frequency')
                app_lengths = np.array([22., 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35])
                app_freqs = np.array([0.001, 0.003, 0.012, 0.034, 0.078, 0.136, 0.182, 0.198, 0.155, 0.104, 0.055, 0.024, 0.017])
                ax6.plot(app_lengths, app_freqs, color='C5', alpha=0.3)

                for ax in (ax1, ax2):
                    for number in learning_rate_halved_at:
                        ax.axvline(number, color='black')

                fgv1 = progress_info[-1][9]
                p1, p2 = fgv1
                fgv1_xs = np.arange(len(p1[:, 0])) / 2. + 1
                ax7.plot(fgv1_xs, p1[:, 0], color='C0', label='player 1')
                # ax7.plot(p2[:, 1], color='C0', alpha=0.4)
                ax7.plot(fgv1_xs, p2[:, 0], color='C1', label='player 2')
                # ax7.plot(p1[:, 1], color='C1', alpha=0.4)
                ax7.set_ylim(0, 1)
                ax7.grid()
                ax7.legend()
                ax7.set_xlabel('round')
                ax7.set_ylabel('player move probabilities')

                fgv1 = progress_info[-1][10]
                p1, p2 = fgv1
                fgv1_xs = np.arange(len(p1[:, 0])) / 2. + 1
                ax8.plot(fgv1_xs, p1[:, 0], color='C0', label='player 1')
                # ax8.plot(p2[:, 1], color='C0', alpha=0.4)
                ax8.plot(fgv1_xs, p2[:, 0], color='C1', label='player 2')
                # ax8.plot(p1[:, 1], color='C1', alpha=0.4)
                ax8.set_ylim(0, 1)
                ax8.grid()
                ax8.legend()
                ax8.set_xlabel('round')
                ax8.set_ylabel('player move probabilities')

                fgv1 = progress_info[-1][11]
                p1, p2 = fgv1
                fgv1_xs = np.arange(len(p1[:, 0])) / 2. + 1
                ax9.plot(fgv1_xs, p1[:, 0], color='C0', label='player 1')
                # ax9.plot(p2[:, 1], color='C0', alpha=0.4)
                ax9.plot(fgv1_xs, p2[:, 0], color='C1', label='player 2')
                # ax9.plot(p1[:, 1], color='C1', alpha=0.4)
                ax9.set_ylim(0, 1)
                ax9.grid()
                ax9.legend()
                ax9.set_xlabel('round')
                ax9.set_ylabel('player move probabilities')

                for index, ax in zip((13, 14, 15), (ax10, ax11, ax12)):
                    scores = progress_info[-1][index]
                    p1, p2 = scores
                    xs = np.arange(len(p1)) / 2. + 1
                    ax.plot(xs, p1, color='C0', label='player 1')
                    ax.plot(xs, p2, color='C1', label='player 2')
                    ax.set_ylim(0, 20)
                    ax.set_yticks(np.arange(0, 21, 2))
                    ax.axhline(15, color='black', linewidth=2)
                    ax.set_xlabel('round')
                    ax.set_ylabel('player score')
                    ax.grid()
                    ax.legend()

                for index, ax in zip((16, 17, 18), (ax13, ax14, ax15)):
                    cards_played = progress_info[-1][index]
                    num_gems_in_supply = progress_info[-1][index + 3]
                    p1, p2 = cards_played
                    xs = np.arange(len(p1)) / 2. + 1
                    ax.plot(xs, p1[:, 0], color='C2', label='P1 T1')
                    ax.plot(xs, p1[:, 1], color='C3', label='P1 T2')
                    ax.plot(xs, p1[:, 2], color='C4', label='P1 T3')
                    ax.plot(xs, p2[:, 0], '--', color='C2', label='P2 T1')
                    ax.plot(xs, p2[:, 1], '--', color='C3', label='P2 T2')
                    ax.plot(xs, p2[:, 2], '--', color='C4', label='P2 T3')
                    ax.plot(xs, num_gems_in_supply, color='C5', alpha=0.3)
                    ax.set_ylim(0, 18)
                    ax.set_yticks(np.arange(0, 19, 2))
                    ax.set_xlabel('round')
                    ax.set_ylabel('cards played')
                    ax.grid()
                    ax.legend(loc=2)
                
                for ax in (ax16, ax17, ax18):
                    ax.set_axis_off()

                fig.set_size_inches((20, 12))
                fig.tight_layout()
                fig.savefig('output.png')
                print('done')

                print('\nExample game:')
                for i, move_info in enumerate(progress_info[-1][12]):
                    print(i % 2, move_info.move, [m for m in move_info.post_move_values])
                # import ipdb
                # ipdb.set_trace()
                print()

                # import ipdb
                # ipdb.set_trace()
    except KeyboardInterrupt:
        ai.saver.save(ai.session, ai.ckpt_filen())

    from numpy import average, std
    print('Average number of rounds: {} ± {}'.format(average(round_nums), std(round_nums)))

def rolling_average(ps):
    N  = 5
    cumsum = np.cumsum(np.insert(ps, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()
