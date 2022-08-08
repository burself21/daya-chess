import numpy as np
import gym
import random

import sys
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from six import StringIO

from gym import spaces, error, utils
from gym.utils import seeding
from datetime import datetime

import json

from .ChessAgent import ChessAgent, RLAgent, RandomAgent
#from ChessAgent import ChessAgent

piece_indices = {'K': 1, 'Q': 2, 'R': 3, 'B': 4, 'N': 5,
                     'a': 6, 'b': 6, 'c': 6, 'd': 6, 'e': 6, 'f': 6, 
                     'g': 6, 'h': 6}


def str_to_square(square):
    return np.array([8 - int(square[1]), ord(square[0]) - 97 ], dtype=np.int8)
    
def play_move(env, move_str):
    player = env.current_player
    promotion_move_str = ""
    if "=" in move_str:
        move_str, promotion_str = move_str.split('=')
        promotion_piece = ID_TO_TYPE[piece_indices[promotion_str[0]]].upper()
        promotion_move_str = f"PROMOTE_{promotion_piece}"
    if move_str[:7] == 'PROMOTE':
        if move_str in env.possible_moves:
            return env.step(env.move_to_action(move_str))
    elif move_str[:3] == 'O-O':
        if move_str[:5] == 'O-O-O':
            if player == 'WHITE':
                move = "CASTLE_QUEEN_SIDE_WHITE"
            else:
                move = "CASTLE_QUEEN_SIDE_BLACK"
        else:
            if player == 'WHITE':
                move = "CASTLE_KING_SIDE_WHITE"
            else:
                move = "CASTLE_KING_SIDE_BLACK"
        if move in env.possible_moves:
            return env.step(env.move_to_action(move))
    else:
        sign = 2 * env.current_player_is_white - 1
        move_str = move_str.replace('x', '').replace('+', '').replace('#', '')
        piece_index = piece_indices[move_str[0]] * sign
        pos_modifier = move_str[1] if len(move_str) > 3 else move_str[0] if abs(piece_index) == 6 and len(move_str) == 3 else None
        square2 = str_to_square(move_str[-2:])
        possible_moves = np.array(list(filter(lambda s: type(s) != str, env.possible_moves)))
        matching_moves = possible_moves[np.all(possible_moves[:, 1] == square2, axis=1)]
        for start, _ in matching_moves:
            if env.state[tuple(start)] == piece_index:
                if pos_modifier:
                    if pos_modifier.isdigit():
                        if int(pos_modifier) != 8 - start[0]:
                            continue
                    else:
                        if ord(pos_modifier) - 97 != start[1]:
                            continue
                return_stuff = env.step(env.move_to_action([start, square2]))
                if promotion_move_str:
                    return_stuff = play_move(env, promotion_move_str)
                return return_stuff
    print("Move is mistyped or illegal:", move_str + "|")
    return False

EMPTY_SQUARE_ID = 0
KING_ID = 1
QUEEN_ID = 2
ROOK_ID = 3
BISHOP_ID = 4
KNIGHT_ID = 5
PAWN_ID = 6

KING = "king"
QUEEN = "queen"
ROOK = "rook"
BISHOP = "bishop"
KNIGHT = "knight"
PAWN = "pawn"

KING_DESC = "K"
QUEEN_DESC = "Q"
ROOK_DESC = "R"
BISHOP_DESC = "B"
KNIGHT_DESC = "N"
PAWN_DESC = ""

WHITE_ID = 1
BLACK_ID = -1

WHITE = "WHITE"
BLACK = "BLACK"

PAWN_VALUE = 2
KNIGHT_VALUE = 6
BISHOP_VALUE = 7
ROOK_VALUE = 10
QUEEN_VALUE = 18
WIN_REWARD = 10000
DRAW_REWARD = 0
LOSS_REWARD = -10000
BASE_ACTION_REWARD = 0
ILLEGAL_ACTION_REWARD = -10
ILLEGAL_ACTION_PENALTY = -1
ILLEGAL_STOP_EARLY_REWARD = -10
GOOD_STOP_EARLY_REWARD = 10
REPEAT_REWARD = -5


@dataclass
class Piece:
    id: int
    icon: str
    desc: str
    type: str
    color: str
    value: float


PIECES = [
    Piece(icon="♟", desc=PAWN_DESC, color=BLACK, type=PAWN, id=-PAWN_ID, value=PAWN_VALUE),
    Piece(icon="♞", desc=KNIGHT_DESC, color=BLACK, type=KNIGHT, id=-KNIGHT_ID, value=KNIGHT_VALUE),
    Piece(icon="♝", desc=BISHOP_DESC, color=BLACK, type=BISHOP, id=-BISHOP_ID, value=BISHOP_VALUE),
    Piece(icon="♜", desc=ROOK_DESC, color=BLACK, type=ROOK, id=-ROOK_ID, value=ROOK_VALUE),
    Piece(icon="♛", desc=QUEEN_DESC, color=BLACK, type=QUEEN, id=-QUEEN_ID, value=QUEEN_VALUE),
    Piece(icon="♚", desc=KING_DESC, color=BLACK, type=KING, id=-KING_ID, value=0),
    Piece(icon=". ", desc="", color=None, type=None, id=EMPTY_SQUARE_ID, value=0),
    Piece(icon="♔", desc=KING_DESC, color=WHITE, type=KING, id=KING_ID, value=0),
    Piece(icon="♕", desc=QUEEN_DESC, color=WHITE, type=QUEEN, id=QUEEN_ID, value=QUEEN_VALUE),
    Piece(icon="♖", desc=ROOK_DESC, color=WHITE, type=ROOK, id=ROOK_ID, value=ROOK_VALUE),
    Piece(icon="♗", desc=BISHOP_DESC, color=WHITE, type=BISHOP, id=BISHOP_ID, value=BISHOP_VALUE),
    Piece(icon="♘", desc=KNIGHT_DESC, color=WHITE, type=KNIGHT, id=KNIGHT_ID, value=KNIGHT_VALUE),
    Piece(icon="♙", desc=PAWN_DESC, color=WHITE, type=PAWN, id=PAWN_ID, value=PAWN_VALUE),
]

ID_TO_COLOR = {piece.id: piece.color for piece in PIECES}
ID_TO_ICON = {piece.id: piece.icon for piece in PIECES}
ID_TO_TYPE = {piece.id: piece.type for piece in PIECES}
ID_TO_VALUE = {piece.id: piece.value for piece in PIECES}
ID_TO_DESC = {piece.id: piece.desc for piece in PIECES}

# RESIGN = "RESIGN"
CASTLE_KING_SIDE_WHITE = "CASTLE_KING_SIDE_WHITE"
CASTLE_QUEEN_SIDE_WHITE = "CASTLE_QUEEN_SIDE_WHITE"
CASTLE_KING_SIDE_BLACK = "CASTLE_KING_SIDE_BLACK"
CASTLE_QUEEN_SIDE_BLACK = "CASTLE_QUEEN_SIDE_BLACK"
CASTLE_MOVES = [
    CASTLE_KING_SIDE_WHITE,
    CASTLE_QUEEN_SIDE_WHITE,
    CASTLE_KING_SIDE_BLACK,
    CASTLE_QUEEN_SIDE_BLACK,
]

PROMOTE_QUEEN = "PROMOTE_QUEEN"
PROMOTE_ROOK = "PROMOTE_ROOK"
PROMOTE_BISHOP = "PROMOTE_BISHOP"
PROMOTE_KNIGHT = "PROMOTE_KNIGHT"

PROMOTION_MOVES = [
    PROMOTE_QUEEN,
    PROMOTE_ROOK,
    PROMOTE_BISHOP,
    PROMOTE_KNIGHT
]

DEFAULT_BOARD = np.array(
    [
        [-3, -5, -4, -2, -1, -4, -5, -3],
        [-6, -6, -6, -6, -6, -6, -6, -6],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [3, 5, 4, 2, 1, 4, 5, 3],
    ],
    dtype=np.int8,
)


def highlight(string, background="white", color="gray"):
    return utils.colorize(utils.colorize(string, color), background, highlight=True)

# Encoder for Environment Class
class ChessEnvEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# CHESS GYM ENVIRONMENT CLASS
# ---------------------------
class ChessEnv(gym.Env):
    def __init__(
        self,
        player_color=WHITE,
        opponent="random",
        log=False,
        initial_state=DEFAULT_BOARD,
        seed=None,
        max_moves=10000,
        max_illegal_moves=10000,
        max_capture_score=1000,
        random_until=0,
        illegal_allowed=False,
        opponent_model_file=None,
        from_position=False,
        current_player=None
    ):
        self.observation_space = spaces.Box(-6, 6, (8, 8))
        self.action_space = spaces.Discrete(64 * 64 + 4 + 4 + 1)

        # constants
        self.moves_max = max_moves
        self.illegal_moves_max = max_illegal_moves
        self.max_capture_score = max_capture_score
        self.random_until = random_until
        self.log = log
        self.initial_state = initial_state
        self.player = player_color  # define player #
        self.player_2 = self.get_other_player(player_color)
        self.opponent = opponent  # define opponent

        #variables
        self.recent_actions=[]
        self.illegal_allowed = illegal_allowed
        self.model_file = opponent_model_file
        ##

        #
        # Observation + Action spaces
        # ---------------------------
        #  Observations: 8x8 board with 6 types of pieces for each player + empty square
        #  Actions: (every board position) x (every board position), 4 castles and promotions
        #
        # Note: not every action is legal
        #
        
        if not from_position:
            self.white_to_promote = False
            self.black_to_promote = False

            self.white_won = False
            self.black_won = False
            self.draw = False
            
            self.num_illegal_moves = 0
            
            self.capture_score = 0
            
            self.consecutive_fiddle_moves = 0
            self.can_en_passant = {}
        
        # reset and build state
        self.seed(seed=seed)
        self.rand_seed = seed
        if from_position:
            self.state = initial_state
            self.current_player = current_player
        else:
            self.reset()

    @staticmethod
    def read_json(json_string):
        data = json.loads(json_string)
        env = ChessEnv(player_color=data['player'], opponent=data['opponent'], seed=data.get('rand_seed'), opponent_model_file=data.get('opponent_model_file'),
                      from_position=True, current_player=data['current_player'], initial_state=np.array(data['state'], dtype=np.int8))
        # set additional state variables
        env.white_to_promote = data['white_to_promote']
        env.black_to_promote = data['black_to_promote']

        env.white_won = data['white_won']
        env.black_won = data['black_won']
        env.draw = data['draw']
        
        env.num_illegal_moves = data['num_illegal_moves']
        
        env.capture_score = data['capture_score']
        
        env.consecutive_fiddle_moves = data['consecutive_fiddle_moves']
        env.can_en_passant = data.get('can_en_passant')

        # other state usually set in reset
        env.initial_state = DEFAULT_BOARD
        env.saved_states = defaultdict(lambda: 0, data.get('saved_states') or {})
        env.move_count = data['move_count']
        env.done = data['done']
        env.white_king_castle_possible = data['white_king_castle_possible']
        env.white_queen_castle_possible = data['white_queen_castle_possible']
        env.black_king_castle_possible = data['black_king_castle_possible']
        env.black_queen_castle_possible = data['black_queen_castle_possible']
        
        env.white_king_on_the_board = len(np.where(env.state == KING_ID)[0]) != 0
        env.black_king_on_the_board = len(np.where(env.state == -KING_ID)[0]) != 0

        env.possible_moves = env.get_possible_moves()

        return env

    def to_dict(self):
        dictionary = self.__dict__.copy()
        for key in ['action_space', 'observation_space', 'np_random', 'opponent_policy', 'initial_state', '_possible_moves']:
            dictionary.pop(key, None)
        return dictionary

    def to_json(self):
        '''
        convert the instance of this class to json
        '''
        return json.dumps(self.to_dict(), cls=ChessEnvEncoder)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == "random":
                self.opponent_policy = RandomAgent(None)
            elif self.opponent.lower() == "rl":
                # we have a RL Agent
                if self.model_file:
                    self.opponent_policy = RLAgent(self.model_file)
                else:
                    raise error.Error("RL Agent opponent policy is requested without passing model file.")
            elif self.opponent.lower() == "ffnn":
                self.opponent_policy = ChessAgent("ffnn", "model8_2_2.pt", max_depth=2)
            elif self.opponent.lower() == "cnn":
                self.opponent_policy = ChessAgent("cnn", "cnn_50k_109.pt", max_depth=2)
            elif self.opponent == "none":
                self.opponent_policy = None
            else:
                raise error.Error(f"Unrecognized opponent policy {self.opponent}")
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs -> observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.state = self.initial_state
        self.done = False
        self.current_player = WHITE
        self.saved_states = defaultdict(lambda: 0)
        self.move_count = 0
        self.white_king_castle_possible = True
        self.white_queen_castle_possible = True
        self.black_king_castle_possible = True
        self.black_queen_castle_possible = True
        self.capture_score = 0
        
        self.white_to_promote = False
        self.black_to_promote = False

        self.white_won = False
        self.black_won = False
        self.draw = False
        
        self.num_illegal_moves = 0
        self.consecutive_fiddle_moves = 0
        self.can_en_passant = {}
        self.recent_actions = []
        
        self.white_king_on_the_board = len(np.where(self.state == KING_ID)[0]) != 0
        self.black_king_on_the_board = len(np.where(self.state == -KING_ID)[0]) != 0
        self.possible_moves = self.get_possible_moves(state=self.state, player=WHITE)
        # If player chooses black, make white openent move first
        if self.player == BLACK:
            white_first_action = self.opponent_policy.get_action(self.get_opponent_env())
            # make move
            # self.state, _, _, _ = self.step(white_first_action)
            self.state, _, _, _ = self.player_move(white_first_action)
            self.move_count += 1
            self.current_player = BLACK
            self.possible_moves = self.get_possible_moves(state=self.state, player=BLACK)
        
        if self.random_until > 0:
            done = False
            if self.opponent_policy:
                done = self.step_n_random(self.random_until)
            else:
                done = self.step_n_random(2 * self.random_until)
            if done:
                return self.reset()
        return self.state

    def copy(self):
        env = copy(self)
        env.saved_states = copy(self.saved_states)
        return env

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        # validate action
        assert self.action_space.contains(action), "ACTION ERROR {}".format(action)

        reward = 0

        # action invalid in current state
        if action not in self.possible_actions:
            if self.illegal_allowed:
                reward += ILLEGAL_ACTION_PENALTY
                action = min(self.possible_actions, key=lambda x: abs(action - x))
            else:
                reward = ILLEGAL_ACTION_REWARD
                self.num_illegal_moves += 1
                if self.num_illegal_moves >= self.illegal_moves_max:
                    self.done = True
                    reward += ILLEGAL_STOP_EARLY_REWARD
                return self.state, reward, self.done, {}#self.info

        # Game is done
        if self.done:
            return (
                self.state,
                0.0,
                True,
                {}, #self.info
            )

        # valid action reward
        reward += BASE_ACTION_REWARD
        # make move
        self.state, move_reward, self.done, must_promote = self.player_move(action)
        reward += move_reward
        
        if must_promote:
            self.possible_moves = self.get_possible_moves()
            return self.state, reward, self.done, {} #self.info
        
        # opponent play
        opponent_player = self.switch_player()
        
        # If we just had 50-move or insufficient material
        #print(move_reward, self.done)
        if self.done and move_reward == DRAW_REWARD:
            return self.state, reward, self.done, {} #self.info

        # 3-fold repetition => DRAW
        encoded_state = self.encode_state()
        self.saved_states[encoded_state] += 1
        if self.saved_states[encoded_state] >= 3:
            self.done = True
            self.draw = True
            print("DRAW BY THREEFOLD.")
            return self.state, DRAW_REWARD, True, {}
        
        self.possible_moves = self.get_possible_moves(player=opponent_player)
        # check if there are no possible_moves for opponent
        if not self.possible_moves:
            self.done = True
            if self.king_is_checked(
                state=self.state, player=opponent_player
            ):
                reward = WIN_REWARD
                print(f"CHECKMATE! {self.get_other_player(opponent_player)} WINS!")
                self.white_won = self.current_player_is_black
                self.black_won = self.current_player_is_white
            else:
                reward = DRAW_REWARD
                print("STALEMATE!")
                self.draw = True
        # if we have max capture score
        elif self.capture_score >= self.max_capture_score:
            self.done = True
            reward += GOOD_STOP_EARLY_REWARD
        if self.done:
            return self.state, reward, self.done, {} # self.info

        # Bot Opponent play
        if self.opponent_policy:
            opponent_action = self.opponent_policy.get_action(self.get_opponent_env())
            # make move
            self.state, opp_reward, self.done, must_promote = self.player_move(opponent_action)
            reward -= opp_reward
            # check if we must promote
            if must_promote:
                self.possible_moves = self.get_possible_moves(player=opponent_player)
                opponent_action = self.opponent_policy.get_action(self.get_opponent_env())
                self.state, opp_reward, self.done, _ = self.player_move(opponent_action)
                reward -= opp_reward
            # increment count on WHITE
            if self.current_player == WHITE:
                self.move_count += 1
            agent_player = self.switch_player()

            # If we just had 50-move or insufficient material
            if self.done and opp_reward == DRAW_REWARD:
                return self.state, reward, self.done, {}

            # 3-fold repetition => DRAW
            encoded_state = self.encode_state()
            self.saved_states[encoded_state] += 1
            if self.saved_states[encoded_state] >= 3:
                self.done = True
                self.draw = True
                print("DRAW BY THREEFOLD.")
                return self.state, DRAW_REWARD, True, {}

            self.possible_moves = self.get_possible_moves(player=agent_player)
            # check if there are no possible_moves for opponent
            if not self.possible_moves:
                self.done = True
                if self.king_is_checked(
                    state=self.state, player=agent_player
                ):
                    reward += LOSS_REWARD
                    print(f"CHECKMATE! {opponent_player} WINS!")
                    self.white_won = self.current_player_is_black
                    self.black_won = self.current_player_is_white
                else:
                    reward += DRAW_REWARD
                    print("STALEMATE!")
                    self.draw = True
            
        elif self.current_player == BLACK:
            self.move_count += 1
        
        if self.move_count >= self.moves_max:
            return (
                self.state,
                reward,
                True,
                {},   #self.info
        )

        return self.state, reward, self.done, {} #self.info
    
    def step_n_random(self, n):
        for _ in range(n):
            action = self.possible_actions[np.random.choice(np.arange(len(self.possible_actions)))]
            _, _, done, _ = self.step(action)
            if done:
                return True
        return False

    def switch_player(self):
        other_player = self.get_other_player(self.current_player)
        self.current_player = other_player
        return other_player

    @property
    def possible_moves(self):
        return self._possible_moves

    @possible_moves.setter
    def possible_moves(self, moves):
        self._possible_moves = moves

    @property
    def possible_actions(self):
        return [self.move_to_action(m) for m in self.possible_moves]

    @property
    def info(self):
        return dict(
            state=self.state,
            move_count=self.move_count,
        )

    @property
    def opponent_player(self):
        if self.current_player == WHITE:
            return BLACK
        return WHITE

    @property
    def current_player_is_white(self):
        return self.current_player == WHITE

    @property
    def current_player_is_black(self):
        return not self.current_player_is_white

    def player_can_castle(self, player):
        if player == WHITE:
            return self.white_king_castle_possible or self.white_queen_castle_possible
        else:
            return self.black_king_castle_possible or self.black_queen_castle_possible

    def get_other_player(self, player):
        if player == WHITE:
            return BLACK
        return WHITE

    def player_move(self, action):
        """
        Returns (state, reward, done, must_promote)
        """
        # Resign
        #if self.is_resignation(action):
        #    return self.state, LOSS_REWARD, True
        # Play
        move = self.action_to_move(action)
        new_state, reward, must_promote = self.next_state(self.state, self.current_player, move, commit=True)
        if self.current_player == self.player and action in self.recent_actions:
            reward += REPEAT_REWARD
        self.recent_actions.append(int(action))
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)

        # Insufficient Material => DRAW (cases: K-K, K-K-B, K-K-N)
        remaining_pieces = new_state[np.nonzero(new_state)]
        if len(remaining_pieces) <= 3 and np.abs(remaining_pieces).sum() in (2, 6, 7):
            self.draw = True
            print("DRAW BY INSUFFICIENT MATERIAL.")
            return new_state, DRAW_REWARD, True, False
        
        # 50-move rule => DRAW
        ## this might not work, need to make sure works when playing against bot
        if self.consecutive_fiddle_moves >= 100:
            self.draw
            print("DRAW BY 50 MOVE RULE.")
            return new_state, DRAW_REWARD, True, False
        
        
        # Render
        if self.log:
            print(" " * 10, ">" * 10, self.current_player)
            self.render_moves([move], mode="human")
        if type(move) is str and move in PROMOTION_MOVES:
            self.white_to_promote = False
            self.black_to_promote = False
        return new_state, reward, False, must_promote

    def next_state(self, state, player, move, commit=False):
        """
        Return the next state given a move
        -------
        (next_state, reward)
        """
        new_state = copy(state)
        reward = 0
        must_promote = False

        if type(move) is str:
            if move in CASTLE_MOVES:
                self.run_castle_move(new_state, move, commit=commit)
                if commit:
                    self.consecutive_fiddle_moves += 1
            elif move in PROMOTION_MOVES:
                reward += self.run_promotion_move(new_state, move, player)
                reward -= BASE_ACTION_REWARD
                #if commit:
                #    self.white_to_promote = False
                #    self.black_to_promote = False
        elif not type(move) is str:
            # Classic move
            _from, _to = move
            
            en_passant = False
            piece_to_move = new_state[_from[0], _from[1]]
            captured_piece = new_state[_to[0], _to[1]]
            if not captured_piece and ID_TO_TYPE[piece_to_move] == PAWN and _to[1] - _from[1] != 0:
                # pawn capture move without a piece on the diagonal implies en passant
                captured_piece = new_state[_from[0], _to[1]]
                en_passant = True
            assert piece_to_move, f"Bad move: {move} - piece is empty"
            new_state[_from[0], _from[1]] = 0
            new_state[_to[0], _to[1]] = piece_to_move
            if en_passant:
                new_state[_from[0], _to[1]] = 0
            
            # editing environment vars if commit
            if commit:
                # check if pawn move or capture
                if ID_TO_TYPE[piece_to_move] == PAWN or captured_piece != EMPTY_SQUARE_ID:
                    self.consecutive_fiddle_moves = 0
                else:
                    self.consecutive_fiddle_moves += 1
                    #print(piece_to_move, ID_TO_TYPE[piece_to_move], self.consecutive_fiddle_moves)     

                self.can_en_passant = {}
            
                if ID_TO_TYPE[piece_to_move] == PAWN:

                    # Pawn double-move (allows other side to en-passant)
                    if abs(_to[0] - _from[0]) == 2:
                        if _from[1] > 0:
                            self.can_en_passant[str(_from[1] - 1)] = int(_from[1])
                        if _from[1] < 7:
                            self.can_en_passant[str(_from[1] + 1)] = int(_from[1])

                    # Pawn reaches end of board
                    elif (player == WHITE and _to[0] == 0) or (player == BLACK and _to[0] == 7):
                        must_promote = True
                        self.file_to_promote = int(_to[1])
                        if player == WHITE:
                            self.white_to_promote = True
                        else:
                            self.black_to_promote = True

                # Keep track if castling is still possible
                if commit and self.player_can_castle(player):
                    if ID_TO_TYPE[piece_to_move] == KING:
                        if player == WHITE:
                            self.white_king_castle_possible = False
                            self.white_queen_castle_possible = False
                        else:
                            self.black_king_castle_possible = False
                            self.black_queen_castle_possible = False
                    elif ID_TO_TYPE[piece_to_move] == ROOK:
                        if _from[1] == 0:
                            if player == WHITE:
                                self.white_queen_castle_possible = False
                            else:
                                self.black_queen_castle_possible = False
                        elif _from[1] == 7:
                            if player == WHITE:
                                self.white_king_castle_possible = False
                            else:
                                self.black_king_castle_possible = False

            # Reward
            reward += ID_TO_VALUE[captured_piece]
            if commit and player == self.player:
                self.capture_score += ID_TO_VALUE[captured_piece]

        return new_state, reward, must_promote
    
    def run_promotion_move(self, state, move, player, commit=False):
        if move == "PROMOTE_QUEEN":
            piece_id = QUEEN_ID
        elif move == "PROMOTE_ROOK":
            piece_id = ROOK_ID
        elif move == "PROMOTE_BISHOP":
            piece_id = BISHOP_ID
        else:
            piece_id = KNIGHT_ID
    
        if player == WHITE:
            state[0, self.file_to_promote] = piece_id
        else:
            state[7, self.file_to_promote] = -piece_id
        return piece_id
        
    def run_castle_move(self, state, move, commit=False):
        if move == CASTLE_KING_SIDE_WHITE:
            state[7, 4] = EMPTY_SQUARE_ID
            state[7, 5] = ROOK_ID
            state[7, 6] = KING_ID
            state[7, 7] = EMPTY_SQUARE_ID
        elif move == CASTLE_QUEEN_SIDE_WHITE:
            state[7, 0] = EMPTY_SQUARE_ID
            state[7, 1] = EMPTY_SQUARE_ID
            state[7, 2] = KING_ID
            state[7, 3] = ROOK_ID
            state[7, 4] = EMPTY_SQUARE_ID
        elif move == CASTLE_KING_SIDE_BLACK:
            state[0, 4] = EMPTY_SQUARE_ID
            state[0, 5] = -ROOK_ID
            state[0, 6] = -KING_ID
            state[0, 7] = EMPTY_SQUARE_ID
        elif move == CASTLE_QUEEN_SIDE_BLACK:
            state[0, 0] = EMPTY_SQUARE_ID
            state[0, 1] = EMPTY_SQUARE_ID
            state[0, 2] = -KING_ID
            state[0, 3] = -ROOK_ID
            state[0, 4] = EMPTY_SQUARE_ID
        if commit:
            if self.current_player_is_white:
                self.white_king_castle_possible = False
                self.white_queen_castle_possible = False
            else:
                self.black_king_castle_possible = False
                self.black_queen_castle_possible = False

    def state_to_grid(self):
        grid = [[f" {ID_TO_ICON[square]} " for square in row] for row in self.state]
        return grid

    def render_grid(self, grid, mode="human"):
        outfile = sys.stdout if mode == "human" else StringIO()
        outfile.write("    ")
        outfile.write("-" * 35)
        outfile.write("\n")
        rows = "87654321"
        for i, row in enumerate(grid):
            outfile.write(f" {rows[i]} | ")
            for square in row:
                outfile.write(square)
            outfile.write(" |\n")
        outfile.write("    ")
        outfile.write("-" * 33)
        outfile.write("\n      a   b   c   d   e   f   g   h ")
        outfile.write("\n")

        if mode == "string":
            return outfile.getvalue()
        if mode != "human":
            return outfile

    def render(self, mode="human"):
        """Render the playing board"""
        grid = self.state_to_grid()
        out = self.render_grid(grid, mode=mode)
        return out

    def render_moves(self, moves, mode="human"):
        grid = self.state_to_grid()
        for move in moves:
            if type(move) is str and move != "RESIGN":
                if move in PROMOTION_MOVES:
                    if self.white_to_promote:
                        grid[0][self.file_to_promote] = highlight(grid[0][self.file_to_promote], background="blue")
                    elif self.black_to_promote:
                        grid[7][self.file_to_promote] = highlight(grid[0][self.file_to_promote], background="blue")
                else:
                    if move == CASTLE_QUEEN_SIDE_WHITE:
                        grid[7][0] = highlight(grid[7][0], background="white")
                        grid[7][1] = highlight(" >>", background="green")
                        grid[7][2] = highlight("> <", background="green")
                        grid[7][3] = highlight("<< ", background="green")
                        grid[7][4] = highlight(grid[7][4], background="white")
                    elif move == CASTLE_KING_SIDE_WHITE:
                        grid[7][4] = highlight(grid[7][4], background="white")
                        grid[7][5] = highlight(" >>", background="green")
                        grid[7][6] = highlight("<< ", background="green")
                        grid[7][7] = highlight(grid[7][7], background="white")
                    elif move == CASTLE_QUEEN_SIDE_BLACK:
                        grid[0][0] = highlight(grid[0][0], background="white")
                        grid[0][1] = highlight(" >>", background="green")
                        grid[0][2] = highlight("> <", background="green")
                        grid[0][3] = highlight("<< ", background="green")
                        grid[0][4] = highlight(grid[0][4], background="white")
                    elif move == CASTLE_KING_SIDE_BLACK:
                        grid[0][4] = highlight(grid[0][4], background="white")
                        grid[0][5] = highlight(" >>", background="green")
                        grid[0][6] = highlight("<< ", background="green")
                        grid[0][7] = highlight(grid[0][7], background="white")
                continue

            x0, y0 = move[0][0], move[0][1]
            x1, y1 = move[1][0], move[1][1]
            if len(grid[x0][y0]) <= 4:
                grid[x0][y0] = highlight(grid[x0][y0], background="white")
            if len(grid[x1][y1]) <= 4:
                if self.state[x1, y1]:
                    grid[x1][y1] = highlight(grid[x1][y1], background="red")
                else:
                    grid[x1][y1] = highlight(grid[x1][y1], background="green")
                    if ID_TO_TYPE[self.state[x0, y0]] == PAWN and y0 != y1:
                        grid[x0][y1] = highlight(grid[x0][y1], background="red")
        return self.render_grid(grid, mode=mode)

    def move_to_action(self, move):
        if not type(move) is str:
            _from = move[0][0] * 8 + move[0][1]
            _to = move[1][0] * 8 + move[1][1]
            return _from * 64 + _to
        if move == CASTLE_KING_SIDE_WHITE:
            return 64 * 64
        elif move == CASTLE_QUEEN_SIDE_WHITE:
            return 64 * 64 + 1
        elif move == CASTLE_KING_SIDE_BLACK:
            return 64 * 64 + 2
        elif move == CASTLE_QUEEN_SIDE_BLACK:
            return 64 * 64 + 3
        elif move == PROMOTE_QUEEN:
            return 64 * 64 + 4
        elif move == PROMOTE_ROOK:
            return 64 * 64 + 5
        elif move == PROMOTE_BISHOP:
            return 64 * 64 + 6
        elif move == PROMOTE_KNIGHT:
            return 64 * 64 + 7
        #elif move == RESIGN:
        #    return 64 * 64 + 8

    def action_to_move(self, action):
        if action >= 64 * 64:
            _action = action - 64 * 64
            if _action == 0:
                return CASTLE_KING_SIDE_WHITE
            elif _action == 1:
                return CASTLE_QUEEN_SIDE_WHITE
            elif _action == 2:
                return CASTLE_KING_SIDE_BLACK
            elif _action == 3:
                return CASTLE_QUEEN_SIDE_BLACK
            elif _action == 4:
                return PROMOTE_QUEEN
            elif _action == 5:
                return PROMOTE_ROOK
            elif _action == 6:
                return PROMOTE_BISHOP
            elif _action == 7:
                return PROMOTE_KNIGHT
            elif _action == 8:
                return self.action_to_move(65)
        _from, _to = action // 64, action % 64
        x0, y0 = _from // 8, _from % 8
        x1, y1 = _to // 8, _to % 8
        return np.array([np.array([x0, y0], dtype=np.int8), np.array([x1, y1], dtype=np.int8)])

    def move_to_string(self, move):
        if type(move) is str:
            if move in [CASTLE_KING_SIDE_WHITE, CASTLE_KING_SIDE_BLACK]:
                return "O-O"
            elif move in [CASTLE_QUEEN_SIDE_WHITE, CASTLE_QUEEN_SIDE_BLACK]:
                return "O-O-O"
            elif move in PROMOTION_MOVES:
                return move
        _from, _to = move
        rows = list(reversed("12345678"))
        cols = "abcdefgh"
        piece_id = self.state[_from[0], _from[1]]
        piece_desc = ID_TO_DESC[piece_id]
        capture = self.state[_to[0], _to[1]] != 0
        _from_str = cols[_from[1]] + rows[_from[0]]
        _to_str = cols[_to[1]] + rows[_to[0]]
        string = f"{piece_desc}{_from_str}{'x' if capture else ''}{_to_str}"
        return string

    def get_possible_actions(self):
        moves = self.get_possible_moves(player=self.current_player)
        return [self.move_to_action(move) for move in moves]

    def get_possible_moves(self, state=None, player=None, attack=False, skip_pawns=False):
        if state is None:
            state = self.state
        if player is None:
            player = self.current_player
        
        if self.white_to_promote or self.black_to_promote:
            return PROMOTION_MOVES
        squares_under_attack = []
        if not attack:
            opponent_player = self.get_other_player(player)
            squares_under_attack = self.get_squares_attacked_by_player(state, opponent_player)

        squares_under_attack_hashmap = defaultdict(lambda: None)
        for sq in squares_under_attack:
            squares_under_attack_hashmap[tuple(sq)] = True

        moves = []
        for coords, piece_id in np.ndenumerate(state):
            coords = np.array(coords, dtype=np.int8)
            if piece_id == 0:
                continue
            color = ID_TO_COLOR[piece_id]
            if color != player:
                continue
            piece_type = ID_TO_TYPE[piece_id]
            if piece_type == KING:
                moves += self.king_moves(
                    player,
                    coords,
                    state=state,
                    attack=attack,
                    squares_under_attack_hashmap=squares_under_attack_hashmap,
                )
            elif piece_type == QUEEN:
                moves += self.queen_moves(player, coords, state=state, attack=attack)
            elif piece_type == ROOK:
                moves += self.rook_moves(player, coords, state=state, attack=attack)
            elif piece_type == BISHOP:
                moves += self.bishop_moves(player, coords, state=state, attack=attack)
            elif piece_type == KNIGHT:
                moves += self.knight_moves(player, coords, state=state, attack=attack)
            elif piece_type == PAWN and not skip_pawns:
                moves += self.pawn_moves(player, coords, state=state, attack=attack)

        if attack:
            return moves

        if self.player_can_castle(player):
            moves += self.castle_moves(
                player, state=state, squares_under_attack_hashmap=squares_under_attack_hashmap
            )

        # Filter out moves that leave the king checked / pass through check for castling
        def move_leaves_king_checked(move):
            # skip castling moves
            #if type(move) is not list:
            #    return False
            # skip king moves
            #if (player == WHITE and state[move[0][0], move[0][1]] == KING_ID) or (
            #    player == BLACK and state[move[0][0], move[0][1]] == -KING_ID
            #):
            #    return False
            next_state, _, _ = self.next_state(state, player, move, commit=False)
            return self.king_is_checked(state=next_state, player=player)
        
        # print(moves)
        moves = [move for move in moves if not move_leaves_king_checked(move)]
        return moves

    def king_moves(
        self,
        player,
        coords,
        state=None,
        attack=False,
        squares_under_attack_hashmap=defaultdict(lambda: None),
    ):
        """KING MOVES"""
        if state is None:
            state = self.state
        moves = []
        steps = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

        if attack:
            for step in steps:
                square = coords + np.array(step, dtype=np.int8)
                if self.king_attack(player, state, square):
                    moves.append(np.array([coords, square]))
        else:
            for step in steps:
                square = coords + np.array(step, dtype=np.int8)
                if self.king_move(player, state, square, squares_under_attack_hashmap):
                    moves.append(np.array([coords, square]))
        return moves

    def queen_moves(self, player, coords, state=None, attack=False):
        """QUEEN MOVES"""
        if state is None:
            state = self.state
        moves = []
        moves += self.rook_moves(player, coords, state=state, attack=attack)
        moves += self.bishop_moves(player, coords, state=state, attack=attack)
        return moves

    def rook_moves(self, player, coords, state=None, attack=False):
        """ROOK MOVES"""
        if state is None:
            state = self.state
        moves = []
        for step in [[-1, 0], [+1, 0], [0, -1], [0, +1]]:
            moves += self.iterativesteps(player, state, coords, step, attack=attack)
        return moves

    def bishop_moves(self, player, coords, state=None, attack=False):
        """BISHOP MOVES"""
        if state is None:
            state = self.state
        moves = []
        for step in [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]:
            moves += self.iterativesteps(player, state, coords, step, attack=attack)
        return moves

    def iterativesteps(self, player, state, coords, step, attack=False):
        """Used to calculate Bishop, Rook and Queen moves"""
        moves = []
        k = 1
        step = np.array(step, dtype=np.int8)
        while True:
            square = coords + k * step
            if attack:
                add_bool, stop_bool = self.attacking_move(player, state, square)
                if add_bool:
                    moves.append(np.array([coords, square]))
                if stop_bool:
                    break
                else:
                    k += 1
            else:
                add_bool, stop_bool = self.playable_move(player, state, square)
                if add_bool:
                    moves.append(np.array([coords, square]))
                if stop_bool:
                    break
                else:
                    k += 1
        return moves

    def knight_moves(self, player, coords, state=None, attack=False):
        """KNIGHT MOVES"""
        if state is None:
            state = self.state
        moves = []
        steps = [
            [-2, -1],
            [-2, +1],
            [+2, -1],
            [+2, +1],
            [-1, -2],
            [-1, +2],
            [+1, -2],
            [+1, +2],
        ]
        # filter:
        for step in steps:
            square = coords + np.array(step, dtype=np.int8)
            if attack:
                is_playable, _ = self.attacking_move(player, state, square)
                if is_playable:
                    moves.append(np.array([coords, square]))
            else:
                is_playable, _ = self.playable_move(player, state, square)
                if is_playable:
                    moves.append(np.array([coords, square]))
        return moves

    def pawn_moves(self, player, coords, state=None, attack=False):
        """PAWN MOVES"""
        if state is None:
            state = self.state
        moves = []
        player_int = ChessEnv.player_to_int(player)
        attack_squares = [
            coords + np.array([1, -1], dtype=np.int8) * (-player_int),
            coords + np.array([1, +1], dtype=np.int8) * (-player_int),
        ]
        one_step_square = coords + np.array([1, 0], dtype=np.int8) * (-player_int)
        two_step_square = coords + np.array([2, 0], dtype=np.int8) * (-player_int)

        if attack:
            for square in attack_squares:
                if ChessEnv.square_is_on_board(square) and not self.is_king_from_player(
                    player, state, square
                ):
                    moves.append(np.array([coords, square]))
        else:
            # moves only to empty squares
            x, y = one_step_square
            if ChessEnv.square_is_on_board(one_step_square) and self.state[x, y] == 0:
                moves.append(np.array([coords, one_step_square]))
            
                # two-step-square only available if 1-step square is available
                x, y = two_step_square
                if ChessEnv.square_is_on_board(two_step_square) and \
                    ((player == WHITE and coords[0] == 6) or (player == BLACK and coords[0] == 1)) and \
                    self.state[x, y] == 0:
                    #
                    moves.append(np.array([coords, two_step_square]))

            # attacks only opponent's pieces
            for square in attack_squares:
                if ChessEnv.square_is_on_board(square) and (self.is_piece_from_other_player(player, state, square) or \
                   (str(coords[1]) in self.can_en_passant and self.can_en_passant[str(coords[1])] == square[1] and \
                       ((player == WHITE and coords[0] == 3) or (player == BLACK and coords[0] == 4)))):
                    moves.append(np.array([coords, square]))

        return moves

    def castle_moves(
        self, player, state=None, squares_under_attack_hashmap=defaultdict(lambda: None)
    ):
        #print(squares_under_attack_hashmap)
        if state is None:
            state = self.state
        moves = []
        if player == WHITE:
            # CASTLE_QUEEN_SIDE_WHITE:
            rook = (7, 0)
            empty_3 = (7, 1)
            empty_2 = (7, 2)
            empty_1 = (7, 3)
            king = (7, 4)
            if (
                self.white_queen_castle_possible
                and state[rook] == ROOK_ID
                and state[empty_3] == EMPTY_SQUARE_ID
                and state[empty_2] == EMPTY_SQUARE_ID
                and state[empty_1] == EMPTY_SQUARE_ID
                and state[king] == KING_ID
                and not squares_under_attack_hashmap[king]
                and not squares_under_attack_hashmap[empty_1]
                and not squares_under_attack_hashmap[empty_2]
            ):
                moves.append(CASTLE_QUEEN_SIDE_WHITE)
            # CASTLE_KING_SIDE_WHITE
            king = (7, 4)
            empty_1 = (7, 5)
            empty_2 = (7, 6)
            rook = (7, 7)
            if (
                self.white_king_castle_possible
                and state[king] == KING_ID
                and state[empty_1] == EMPTY_SQUARE_ID
                and state[empty_2] == EMPTY_SQUARE_ID
                and state[rook] == ROOK_ID
                and not squares_under_attack_hashmap[king]
                and not squares_under_attack_hashmap[empty_1]
                and not squares_under_attack_hashmap[empty_2]
            ):
                moves.append(CASTLE_KING_SIDE_WHITE)
        else:
            # CASTLE_QUEEN_SIDE_BLACK:
            rook = (0, 0)
            empty_3 = (0, 1)
            empty_2 = (0, 2)
            empty_1 = (0, 3)
            king = (0, 4)
            if (
                self.black_queen_castle_possible
                and state[rook] == -ROOK_ID
                and state[empty_3] == EMPTY_SQUARE_ID
                and state[empty_2] == EMPTY_SQUARE_ID
                and state[empty_1] == EMPTY_SQUARE_ID
                and state[king] == -KING_ID
                and not squares_under_attack_hashmap[king]
                and not squares_under_attack_hashmap[empty_1]
                and not squares_under_attack_hashmap[empty_2]
            ):
                moves.append(CASTLE_QUEEN_SIDE_BLACK)
            # CASTLE_KING_SIDE_BLACK:
            king = (0, 4)
            empty_1 = (0, 5)
            empty_2 = (0, 6)
            rook = (0, 7)
            if (
                self.black_king_castle_possible
                and state[king] == -KING_ID
                and state[empty_1] == EMPTY_SQUARE_ID
                and state[empty_2] == EMPTY_SQUARE_ID
                and state[rook] == -ROOK_ID
                and not squares_under_attack_hashmap[king]
                and not squares_under_attack_hashmap[empty_1]
                and not squares_under_attack_hashmap[empty_2]
            ):
                moves.append(CASTLE_KING_SIDE_BLACK)
        return moves

    def king_move(self, player, state, square, squares_under_attack_hashmap):
        """
        return squares to which the king can move,
        i.e. unattacked squares that can be:
        - empty squares
        - opponent pieces (excluding king)
        If opponent king is encountered, then there's a problem...
        => return <bool> is_playable
        """
        if not ChessEnv.square_is_on_board(square):
            return False
        elif squares_under_attack_hashmap[tuple(square)]:
            return False
        elif self.is_piece_from_player(player, state, square):
            return False
        elif self.is_king_from_other_player(player, state, square):
            raise Exception(f"KINGS NEXT TO EACH OTHER ERROR {square}")
        elif self.is_piece_from_other_player(player, state, square):
            return True
        elif state[square[0], square[1]] == 0:  # empty square
            return True
        else:
            raise Exception(f"KING MOVEMENT ERROR {square}")

    def king_attack(self, player, state, square):
        """
        return all the squares that the king can attack, except:
        - squares outside the board
        If opponent king is encountered, then there's a problem...
        => return <bool> is_playable
        """
        if not ChessEnv.square_is_on_board(square):
            return False
        elif self.is_piece_from_player(player, state, square):
            return True
        elif self.is_king_from_other_player(player, state, square):
            raise Exception(f"KINGS NEXT TO EACH OTHER ERROR {square}")
        elif self.is_piece_from_other_player(player, state, square):
            return True
        elif state[square[0], square[1]] == 0:  # empty square
            return True
        else:
            raise Exception(f"KING MOVEMENT ERROR {square}")

    def playable_move(self, player, state, square):
        """
        return squares to which a piece can move
        - empty squares
        - opponent pieces (excluding king)
        => return [<bool> playable, <bool> stop_iteration]
        """
        if not ChessEnv.square_is_on_board(square):
            return False, True
        elif self.is_piece_from_player(player, state, square):
            return False, True
        elif self.is_king_from_other_player(player, state, square):
            return False, True
        elif self.is_piece_from_other_player(player, state, square):
            return True, True
        elif state[square[0], square[1]] == 0:  # empty square
            return True, False
        else:
            print(f"PLAYABLE MOVE ERROR {square}")
            raise Exception(f"PLAYABLE MOVE ERROR {square}")

    def attacking_move(self, player, state, square):
        """
        return squares that are attacked or defended
        - empty squares
        - opponent pieces (opponent king is ignored)
        - own pieces
        => return [<bool> playable, <bool> stop_iteration]
        """
        if not ChessEnv.square_is_on_board(square):
            return False, True
        elif self.is_piece_from_player(player, state, square):
            return True, True
        elif self.is_king_from_other_player(player, state, square):
            return True, True
        elif self.is_piece_from_other_player(player, state, square):
            return True, True
        elif state[square[0], square[1]] == 0:  # empty square
            return True, False
        else:
            print(f"ATTACKING MOVE ERROR {square}")
            raise Exception(f"ATTACKING MOVE ERROR {square}")

    def get_squares_attacked_by_player(self, state, player):
        moves = self.get_possible_moves(state=state, player=player, attack=True)
        attacked_squares = [move[1] for move in moves]
        return attacked_squares

    # def is_current_player_piece(self, square):
    #     self.is_piece_from_player(square, self.current_player)

    # def is_opponent_piece(self, square):
    #     self.is_piece_from_player(square, self.opponent_player)

    def is_piece_from_player(self, player, state, square):
        piece_id = state[square[0], square[1]]
        color = ID_TO_COLOR[piece_id]
        return color == player

    def is_piece_from_other_player(self, player, state, square):
        return self.is_piece_from_player(self.get_other_player(player), state, square)

    # def is_king_from_current_player(self, square):
    #     self.is_king_from_player(square, self.current_player)

    # def is_king_from_opponent_player(self, square):
    #     self.is_king_from_player(square, self.opponent_player)

    def is_king_from_player(self, player, state, square):
        piece_id = state[square[0], square[1]]
        if ID_TO_TYPE[piece_id] != KING:
            return False
        color = ID_TO_COLOR[piece_id]
        return color == player

    def is_king_from_other_player(self, player, state, square):
        return self.is_king_from_player(self.get_other_player(player), state, square)


    @staticmethod
    def player_to_int(player):
        if player == WHITE:
            return 1
        return -1

    @staticmethod
    def square_is_on_board(square):
        return not (square[0] < 0 or square[0] > 7 or square[1] < 0 or square[1] > 7)
    
    @staticmethod
    def encode(state):
        mapping = "0ABCDEFfedcba"
        encoding = "".join([mapping[val] for val in state.ravel()])
        return encoding

    def king_is_checked(self, state=None, player=None):
        if state is None:
            state = self.state
        if player is None:
            player = self.current_player
        # King not present on the board (for testing purposes)
        # if (player == WHITE and not self.white_king_on_the_board) or (
        #     player == BLACK and not self.black_king_on_the_board
        # ):
        #     return False
        player_int = ChessEnv.player_to_int(player)
        king_id = player_int * KING_ID
        king_pos = np.where(state == king_id)
        king_square = [king_pos[0][0], king_pos[1][0]]
        other_player = self.get_other_player(player)
        attacked_squares = self.get_squares_attacked_by_player(state, other_player)
        #print(attacked_squares)
        if not attacked_squares:
            return False
        return any(np.equal(attacked_squares, king_square).all(1))
    
    def get_opponent_env(self):
        opponent_env = self.copy()
        opponent_env.player = self.player_2
        opponent_env.player_2 = self.player
        opponent_env.opponent_policy = None
        return opponent_env

    def encode_state(self):
        return ChessEnv.encode(np.append(self.state, np.array([self.current_player_is_white, self.white_king_castle_possible, self.white_queen_castle_possible,
                                                               self.black_king_castle_possible, self.black_queen_castle_possible
                                                               ], dtype=np.int8)))