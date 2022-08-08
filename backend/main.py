import os
import sys
import random
from uuid import uuid4

from flask import Flask, send_from_directory, make_response, request, redirect, url_for, send_file, session as Session
from flask_cors import CORS, cross_origin
import requests
import json

from chess_nn.ChessEnv import ChessEnv, PROMOTION_MOVES
from chess_nn.ChessAgent import RLAgent

development = 'FLASK_ENV' in os.environ and os.environ['FLASK_ENV'] == 'development'

if not development:
    from google.appengine.api.runtime import runtime
    from google.appengine.api import wrap_wsgi_app

    from google.cloud import firestore



if development:
    app = Flask(__name__)
    db = None
    sessions = None
else:
    app = Flask(__name__, static_url_path='', static_folder='frontend')
    db = firestore.Client()
    sessions = db.collection('sessions')

app.secret_key = "954Fa7FNy3"

# Enable Session
if development:
    app.config["SESSION_PERMANENT"] = True
    app.config["SESSION_TYPE"] = "filesystem"
    app.config['SECRET_KEY'] = "954Fa7FNy3"

#Session(app)

#logging.info(runtime.memory_usage())
#white_agent = RLAgent('rl_white_1')
white_agent = "random"
#logging.info(runtime.memory_usage())
#black_agent = RLAgent('rl_black_1')
black_agent = "random"

#app.wsgi_app = wrap_wsgi_app(app.wsgi_app, use_legacy_context_mode=True, use_deferred=True)


if not development:
    @firestore.transactional
    def get_session_data(transaction, session_id=None, color="WHITE"):
        """ Looks up (or creates) the session with the given session_id.
            Creates a random session_id if none is provided. Increments
            the number of views in this session. Updates are done in a
            transaction to make sure no saved increments are overwritten.
        """
        if session_id is None:
            session_id = str(uuid4())   # Random, unique identifier

        doc_ref = sessions.document(document_id=session_id)
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            session = doc.to_dict()
            session['env'] = ChessEnv.read_json(session['env'])
        else:
            opponent_agent = black_agent if color == "WHITE" else white_agent
            env = ChessEnv(player_color=color, opponent=opponent_agent)
            session = {
                'env': env.to_json()
            }
            transaction.set(doc_ref, session)
            session['env'] = env
        

        session['session_id'] = session_id
        return session

    @firestore.transactional
    def save_session_data(transaction, session_id, env):
        doc_ref = sessions.document(document_id=session_id)
        doc = doc_ref.get(transaction=transaction)
        if not doc.exists:
            print("Error: Session ID Not Found!")
            return False
        else:
            session = {
                'env': env.to_json()
            }
            transaction.set(doc_ref, session)
            return True

#logging.info(runtime.memory_usage())

#Enable CORS
if development:
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['CORS_SUPPORTS_CREDENTIALS'] = True
    # CORS_ALLOW_ORIGIN="*,*"
    # CORS_EXPOSE_HEADERS="*,*"
    # CORS_ALLOW_HEADERS="content-type,*"
    # cors = CORS(app, origins=CORS_ALLOW_ORIGIN.split(","), allow_headers=CORS_ALLOW_HEADERS.split(",") , expose_headers= CORS_EXPOSE_HEADERS.split(","),   supports_credentials = True)
    cors = CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})


environments = {}

# def cond_cross_origin():
#     return lambda fn: (cross_origin(fn) if development else fn)

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    if development:
        Session.permanent = True
        return redirect("http://localhost:4200")
    else:
        return send_file('./frontend/index.html')

@app.route("/get_env", methods=['GET'])
@cross_origin()
def get_env():
    if development:
        if 'id' not in Session:
            # We need a new chess env
            id = os.urandom(12).hex()
            Session['id'] = id
            env = ChessEnv(player_color="WHITE", opponent=black_agent)
            environments[id] = env
        else:
            id = Session['id']
            if id not in environments:
                env = ChessEnv(player_color="WHITE", opponent=black_agent)
                environments[id] = env
            else:
                env = environments[id]
    else:
        session_id = request.cookies.get("session")
        transaction = db.transaction()
        session = get_session_data(transaction, session_id=session_id)
        print(session)
        env = session['env']

    possible_moves = env.possible_moves
    #print(possible_moves)
    response = {'state': env.state.tolist(), 'possible_moves': list(map(lambda move: move if type(move) is str else move.tolist(), possible_moves)),
            'color': env.player}
    if not development:
        response = make_response(response)
        response.set_cookie('session', session['session_id'])
    return response

# @app.route("/reset_env", methods=['GET'])
# @cross_origin()
# def reset_env():
#     hasColor = 'color' in request.args
#     if hasColor:
#         color = request.args.get('color')
#     else:
#         color = "WHITE"
#     opponent_agent = black_agent if color == "WHITE" else white_agent
#     if 'id' not in session:
#         # We need a new chess env
#         id = os.urandom(12).hex()
#         session['id'] = id
#         env = ChessEnv(player_color=color, opponent=opponent_agent)
#         environments[id] = env
#     else:
#         id = session['id']
#         if id not in environments or (hasColor and environments[id].player != color):
#             env = ChessEnv(player_color=color, opponent=opponent_agent)
#             environments[id] = env
#         else:
#             env = environments[id]
#             env.reset()

#     possible_moves = env.possible_moves
#     return {'state': env.state.tolist(), 'possible_moves': list(map(lambda move: move if type(move) is str else move.tolist(), possible_moves))}

@app.route("/new_env", methods=['GET'])
@cross_origin()
def new_env():
    color = request.args.get('color')
    if not color:
        color = "WHITE"

    opponent_agent = black_agent if color == "WHITE" else white_agent
    env = ChessEnv(player_color=color, opponent=opponent_agent)

    session_id = None

    if development:
        if 'id' not in Session:
            # We need a new chess env
            id = os.urandom(12).hex()
            Session['id'] = id
        else:
            id = Session['id']
            environments[id] = env

    else:
        session_id = request.cookies.get("session")
        transaction = db.transaction()
        save_session_data(transaction, session_id, env)


    possible_moves = env.possible_moves
    response = {'state': env.state.tolist(), 'possible_moves': list(map(lambda move: move if type(move) is str else move.tolist(), possible_moves))}
    if not development:
        response = make_response(response)
        response.set_cookie("session", session_id)
    return response

@app.route("/move/<index>", methods=['PUT'])
@cross_origin()
def move(index):
    index = int(index)
    promotion_move = f"PROMOTE_{request.args.get('promote').upper()}" if 'promote' in request.args else ""
    
    if index < 0:
        return {'error': "Invalid move index, must be >= 0"}
    
    # Get environment
    if development:
        if 'id' not in Session:
            # We need a new chess env
            return {"error": "No valid environment found", "environments": json.dumps(environments)}
    
        id = Session['id']
        if id not in environments:
            return {'error': "No valid environment found", "environments": json.dumps(environments)}
        env = environments[id]
    else:
        session_id = request.cookies.get("session")
        transaction = db.transaction()
        session = get_session_data(transaction, session_id=session_id)
        print(session)
        env = session['env']

    if index >= len(env.possible_moves):
        return {'error': "Invalid move index, greater than number of possible moves", "moves": env.possible_moves, "idx": index}

    move = env.possible_moves[index]

    _, reward, done, _ = env.step(env.move_to_action(move))
    if promotion_move in PROMOTION_MOVES and promotion_move in env.possible_moves:
        _, reward, done, _ = env.step(env.move_to_action(promotion_move))
    possible_moves = env.possible_moves
    response = {'state': env.state.tolist(), 'possible_moves': list(map(lambda move: move if type(move) is str else move.tolist(), possible_moves))}
    if done:
        response['result'] = reward // 100
    if not development:
        save_session_data(db.transaction(), session["session_id"], env)
        response = make_response(response)
        response.set_cookie("session", session["session_id"])
    return response

#logging.info(runtime.memory_usage())