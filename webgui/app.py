from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
import time
from functools import lru_cache

# Add paths to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'Tims_splendor_agent'))

from lapidary.game import GameState, Card, Noble
from lapidary.data import colours as colors
from model_adapters import get_adapter, MODEL_ADAPTERS

app = Flask(__name__, static_folder='.', static_url_path='')

# Initialize the model adapter - default to the PPO model for backward compatibility
default_model_path = os.path.join(parent_dir, 'Tims_splendor_agent', 'models', 'ppo_checkpoint_final.pt')
model_adapter = None
current_model_info = {
    "path": default_model_path,
    "type": "ppo",
    "input_dim": 2300,
    "output_dim": 100,
    "status": "Not loaded"
}

# For configuring the model path via environment variable
MODEL_PATH = os.environ.get('SPLENDOR_MODEL_PATH', default_model_path)
MODEL_TYPE = os.environ.get('SPLENDOR_MODEL_TYPE', 'ppo')

try:
    # Try to load the model with the appropriate adapter
    model_adapter = get_adapter(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        input_dim=2300,
        output_dim=100
    )
    model_adapter.load_model()
    current_model_info = {
        "path": MODEL_PATH,
        "type": MODEL_TYPE,
        "input_dim": 2300,
        "output_dim": 100,
        "status": "Loaded"
    }
    print(f"Successfully loaded {MODEL_TYPE} model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    # Set model adapter to None
    model_adapter = None
    current_model_info["status"] = f"Error: {str(e)}"
    print("WARNING: No model is available. The Model Bot will not work until a valid model is loaded.")

# For backward compatibility
ppo_bot = model_adapter

def card_error_handler(func):
    """Decorator to handle errors in card conversion"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            print(f"Arguments: {args}")
            print(f"Keyword arguments: {kwargs}")
            import traceback
            traceback.print_exc()
            raise e
    return wrapper

@card_error_handler
def dict_to_card(card_dict):
    """Convert a card dictionary to a proper Card object"""
    if not isinstance(card_dict, dict):
        return card_dict  # If it's already a Card or None, return as is
    
    # Ensure all required attributes exist
    if 'points' not in card_dict:
        card_dict['points'] = 0
    if 'gems' not in card_dict:
        card_dict['gems'] = {}
    if 'colour' not in card_dict:
        card_dict['colour'] = 'none'  # Default color
    
    # Extract the gem colors from the gems dictionary
    gems = card_dict['gems']
    white = gems.get('white', 0)
    blue = gems.get('blue', 0)
    green = gems.get('green', 0)
    red = gems.get('red', 0)
    black = gems.get('black', 0)
    
    tier = card_dict.get('tier', 1)
    colour = card_dict['colour']
    points = card_dict['points']
    
    # Create a new Card object with the required properties using individual color parameters
    card = Card(
        tier=tier,
        colour=colour,
        points=points,
        white=white,
        blue=blue,
        green=green,
        red=red,
        black=black
    )
    
    return card

@card_error_handler
def dict_to_noble(noble_dict):
    """Convert a noble dictionary to a proper Noble object"""
    if not isinstance(noble_dict, dict):
        return noble_dict  # If it's already a Noble or None, return as is
    
    # Extract card requirements (JavaScript uses cards field, Python uses individual colors)
    cards = noble_dict.get('cards', {})
    white = cards.get('white', 0)
    blue = cards.get('blue', 0)
    green = cards.get('green', 0)
    red = cards.get('red', 0)
    black = cards.get('black', 0)
    
    # In Python Noble constructor, points is first argument with default 3
    points = noble_dict.get('points', 3)
    
    # Create Noble with the extracted properties
    noble = Noble(
        points=points,
        white=white,
        blue=blue,
        green=green,
        red=red,
        black=black
    )
    
    return noble

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/model_move', methods=['POST'])
def model_move():
    """API endpoint for getting a move from any model agent"""
    print("\n==== Model Move Request ====")
    start_time = time.time()
    data = request.json
    
    # Check if model adapter is available
    if model_adapter is None:
        return jsonify({
            'error': 'No model is loaded. Please load a valid model using the "Load Model" button before requesting a move.'
        }), 400
    
    try:
        print(f"Current player in request: {data['current_player_index'] + 1}")
        
        # Create a GameState from the JSON
        game_state = GameState(players=2, init_game=True)
        
        # Set current player
        game_state.current_player_index = data['current_player_index']
        
        # Set up the supply gems
        for color, count in data['supply_gems'].items():
            try:
                setattr(game_state, f'_num_{color}_available', count)
            except AttributeError as e:
                print(f"Warning: Could not set gem {color} to {count}: {e}")
        
        # Set up market cards - making sure we handle card data correctly and convert to Card objects
        try:
            # Convert card dictionaries to proper Card objects
            game_state._tier_1_visible = [dict_to_card(card) for card in data['tier_1_visible']]
            game_state._tier_2_visible = [dict_to_card(card) for card in data['tier_2_visible']]
            game_state._tier_3_visible = [dict_to_card(card) for card in data['tier_3_visible']]
        except Exception as e:
            print(f"Error setting up market cards: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Card setup error: {str(e)}'}), 500
        
        # Set up nobles - convert them to proper Noble objects
        try:
            game_state.nobles = [dict_to_noble(noble) for noble in data['nobles']]
        except Exception as e:
            print(f"Error setting up nobles: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Noble setup error: {str(e)}'}), 500
        
        # Set up players
        for i, player_data in enumerate(data['players']):
            try:
                player = game_state.players[i]
                
                # Set gems
                for color, count in player_data['gems'].items():
                    player.set_gems(color, count)
                
                # Set cards played and reserved - with proper error handling
                try:
                    # Handle cards_played with validation and convert to Card objects
                    valid_cards_played = []
                    for card in player_data.get('cards_played', []):
                        valid_cards_played.append(dict_to_card(card))
                    
                    player.cards_played = valid_cards_played
                    
                    # Handle cards_in_hand with validation and convert to Card objects
                    valid_cards_in_hand = []
                    for card in player_data.get('cards_in_hand', []):
                        valid_cards_in_hand.append(dict_to_card(card))
                    
                    player.cards_in_hand = valid_cards_in_hand
                except Exception as e:
                    print(f"Error setting up cards for player {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Player card setup error: {str(e)}'}), 500
                
                # Set nobles
                try:
                    player.nobles = [dict_to_noble(noble) for noble in player_data.get('nobles', [])]
                except Exception as e:
                    print(f"Error setting up nobles for player {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Player noble setup error: {str(e)}'}), 500
            except Exception as e:
                print(f"Error setting up player {i+1}: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Player setup error: {str(e)}'}), 500
        
        # Get frontend moves - we'll need these regardless
        frontend_moves = data['moves']
        
        # IMPORTANT: Error if there are no valid moves
        if not frontend_moves:
            print("ERROR: No valid moves received from frontend!")
            return jsonify({'error': 'No valid moves available'}), 400
        
        # Classify frontend moves by type for faster matching later
        frontend_moves_by_type = {}
        for i, move in enumerate(frontend_moves):
            move_type = move.get('action')
            if move_type not in frontend_moves_by_type:
                frontend_moves_by_type[move_type] = []
            frontend_moves_by_type[move_type].append((i, move))
        
        # Make sure the game state is valid before getting a move
        try:
            # Limit the number of valid moves to process - makes evaluation faster
            MAX_VALID_MOVES = 30
            valid_moves = game_state.get_valid_moves(game_state.current_player_index)
            
            if len(valid_moves) > MAX_VALID_MOVES:
                print(f"Too many valid moves ({len(valid_moves)}), limiting to {MAX_VALID_MOVES}")
                # Prioritize buy and reserve moves over gem moves
                buy_moves = [m for m in valid_moves if m[0] in ['buy_available', 'buy_reserved']]
                reserve_moves = [m for m in valid_moves if m[0] == 'reserve']
                gem_moves = [m for m in valid_moves if m[0] == 'gems']
                
                valid_moves = []
                # First add buy moves which are most important
                valid_moves.extend(buy_moves[:min(len(buy_moves), MAX_VALID_MOVES // 2)])
                # Then add some reserve moves
                valid_moves.extend(reserve_moves[:min(len(reserve_moves), MAX_VALID_MOVES // 4)])
                # Then add some gem moves to fill up to MAX_VALID_MOVES
                remaining = MAX_VALID_MOVES - len(valid_moves)
                valid_moves.extend(gem_moves[:min(len(gem_moves), remaining)])
            
            if not valid_moves:
                return jsonify({'error': 'No valid moves from game state'}), 400
        except Exception as e:
            print(f"Error getting valid moves: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Valid move calculation error: {str(e)}'}), 500
        
        # Enable caching of state copies to speed up move evaluation
        game_state._cached_copies = {}
        
        # Get moves from the model adapter
        print(f"Getting model move for player {game_state.current_player_index + 1}")
        try:
            # First modify the game state to disable verification to avoid ipdb errors
            game_state._verify_state = False  # Add this attribute to skip verification
            
            # Get the move from the model adapter
            selected_move, confidence = model_adapter.predict_move(game_state, valid_moves)
            print(f"Model selected move: {selected_move} (confidence: {confidence})")
        except Exception as e:
            print(f"Error getting move from model adapter: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Model prediction error: {str(e)}'}), 500
        
        # Find the index of the selected move in the frontend moves list
        move_index = -1
        
        # Fast matching using move type
        move_type = selected_move[0]
        for i, frontend_move in frontend_moves_by_type.get(move_type, []):
            if _moves_match(frontend_move, selected_move):
                move_index = i
                print(f"Found matching move at index {i}")
                break
        
        # If we couldn't find the exact move, use a best-effort approach to find a similar move
        if move_index == -1:
            print("Could not find exact matching move, trying alternative matching approaches")
            
            # Try to match by action type and partial parameters
            for i, frontend_move in frontend_moves_by_type.get(move_type, []):
                if _moves_partially_match(frontend_move, selected_move):
                    move_index = i
                    print(f"Found partially matching move at index {i}")
                    break
        
        # If we still couldn't find a match, return an error
        if move_index == -1:
            # Create a more detailed error message
            move_details = f"Model selected move: {selected_move}"
            
            # Get a list of available move types from frontend_moves
            avail_move_types = {}
            for i, move in enumerate(frontend_moves):
                move_type = move['action']
                if move_type not in avail_move_types:
                    avail_move_types[move_type] = []
                
                # Add a simplified description of the move
                if move_type == 'gems':
                    gems_desc = ", ".join([f"{color}: {count}" for color, count in move['gems'].items() if count > 0])
                    avail_move_types[move_type].append(f"[{i}] {gems_desc}")
                elif move_type == 'reserve':
                    avail_move_types[move_type].append(f"[{i}] tier: {move.get('tier')}, index: {move.get('index')}")
                elif move_type == 'buy_available':
                    avail_move_types[move_type].append(f"[{i}] tier: {move.get('tier')}, index: {move.get('index')}")
                elif move_type == 'buy_reserved':
                    avail_move_types[move_type].append(f"[{i}] index: {move.get('index')}")
            
            # Format the available move types
            move_types_list = []
            for move_type, moves in avail_move_types.items():
                count = len(moves)
                examples = moves[:3]  # Only show up to 3 examples per type
                if count > 3:
                    examples.append(f"... {count-3} more")
                move_types_list.append(f"{move_type} ({count}): {', '.join(examples)}")
            
            error_msg = f"Failed to match model move with frontend moves.\n{move_details}\nAvailable frontend moves:\n" + "\n".join(move_types_list)
            
            return jsonify({'error': error_msg}), 400
        
        elapsed = time.time() - start_time
        print(f"Model move processing took {elapsed:.2f} seconds")
        print("==== Model Move Response Complete ====\n")
        return jsonify({'move_index': move_index})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR in model move handler:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/ppo_move', methods=['POST'])
def ppo_move():
    """Legacy endpoint for backward compatibility - redirects to model_move"""
    return model_move()

def _moves_match(frontend_move, model_move):
    """
    Check if a frontend move exactly matches a model move
    
    Args:
        frontend_move: Move from the frontend
        model_move: Move from the model bot
    
    Returns:
        bool: True if the moves match
    """
    # Check if actions match
    if frontend_move['action'] != model_move[0]:
        return False
    
    action = frontend_move['action']
    
    if action == 'gems':
        # For gem moves, check if gem counts match
        for color in colors:
            frontend_value = frontend_move['gems'].get(color, 0)
            
            # Handle both list and dict formats from model
            if isinstance(model_move[1], dict):
                model_value = model_move[1].get(color, 0)
            else:
                # Try to find color in list by index
                try:
                    color_index = colors.index(color)
                    if color_index < len(model_move[1]):
                        model_value = model_move[1][color_index]
                    else:
                        model_value = 0
                except (ValueError, IndexError):
                    model_value = 0
            
            if frontend_value != model_value:
                return False
        
        return True
    
    elif action == 'reserve':
        # For reserve moves, check tier and index match
        if frontend_move.get('tier') != model_move[1] or frontend_move.get('index') != model_move[2]:
            return False
        
        # Check gold gem gain matches
        frontend_gold = frontend_move.get('gems', {}).get('gold', 0)
        model_gold = 0
        if len(model_move) > 3 and isinstance(model_move[3], dict):
            model_gold = model_move[3].get('gold', 0)
        elif len(model_move) > 3 and isinstance(model_move[3], list) and len(model_move[3]) > 0:
            model_gold = model_move[3][5]  # Assuming gold is at index 5
        
        return frontend_gold == model_gold
    
    elif action == 'buy_available':
        # For buy moves, check tier and index match
        if frontend_move.get('tier') != model_move[1] or frontend_move.get('index') != model_move[2]:
            return False
        
        # Check payment matches
        frontend_payment = frontend_move.get('gems', {})
        model_payment = model_move[3] if len(model_move) > 3 else {}
        
        # Convert model payment to dict if it's a list
        if isinstance(model_payment, list):
            dict_payment = {}
            for i, color in enumerate(colors + ['gold']):
                if i < len(model_payment) and model_payment[i] > 0:
                    dict_payment[color] = model_payment[i]
            model_payment = dict_payment
        
        # Compare payments
        for color in colors + ['gold']:
            frontend_value = frontend_payment.get(color, 0)
            model_value = model_payment.get(color, 0)
            if frontend_value != model_value:
                return False
        
        return True
    
    elif action == 'buy_reserved':
        # For buy reserved moves, check index matches
        if frontend_move.get('index') != model_move[1]:
            return False
        
        # Check payment matches
        frontend_payment = frontend_move.get('gems', {})
        model_payment = model_move[2] if len(model_move) > 2 else {}
        
        # Convert model payment to dict if it's a list
        if isinstance(model_payment, list):
            dict_payment = {}
            for i, color in enumerate(colors + ['gold']):
                if i < len(model_payment) and model_payment[i] > 0:
                    dict_payment[color] = model_payment[i]
            model_payment = dict_payment
        
        # Compare payments
        for color in colors + ['gold']:
            frontend_value = frontend_payment.get(color, 0)
            model_value = model_payment.get(color, 0)
            if frontend_value != model_value:
                return False
        
        return True
    
    return False

def _moves_partially_match(frontend_move, model_move):
    """
    Check if a frontend move partially matches a model move
    Just matches the move type and primary parameters, not the payment
    
    Args:
        frontend_move: Move from the frontend
        model_move: Move from the model bot
    
    Returns:
        bool: True if the moves partially match
    """
    # Check if actions match
    if frontend_move['action'] != model_move[0]:
        return False
    
    action = frontend_move['action']
    
    if action == 'reserve':
        # For reserve moves, just check tier matches
        return frontend_move.get('tier') == model_move[1]
    
    elif action == 'buy_available':
        # For buy moves, just check tier and index match
        return frontend_move.get('tier') == model_move[1] and frontend_move.get('index') == model_move[2]
    
    elif action == 'buy_reserved':
        # For buy reserved moves, just check index matches
        return frontend_move.get('index') == model_move[1]
    
    # For other move types (like gems), use the exact match function
    return _moves_match(frontend_move, model_move)

# Add a new endpoint to configure model path at runtime
@app.route('/api/set_model', methods=['POST'])
def set_model():
    """API endpoint for changing the model at runtime"""
    global model_adapter, current_model_info
    
    data = request.json
    
    if not data or 'model_path' not in data:
        return jsonify({'error': 'Missing model_path parameter'}), 400
        
    model_path = data['model_path']
    model_type = data.get('model_type', 'ppo')  # Default to ppo for backward compatibility
    input_dim = data.get('input_dim', 2300)
    output_dim = data.get('output_dim', 100)
    
    # Validate the model path
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file not found: {model_path}'}), 404
    
    # Validate the model type
    if model_type not in MODEL_ADAPTERS:
        return jsonify({
            'error': f'Unknown model type: {model_type}. Available types: {list(MODEL_ADAPTERS.keys())}'
        }), 400
    
    try:
        # Create a new model adapter with the specified type
        adapter = get_adapter(
            model_type=model_type,
            model_path=model_path,
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # Load the model
        adapter.load_model()
        
        # If successful, update the global model adapter
        model_adapter = adapter
        current_model_info = {
            "path": model_path,
            "type": model_type,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "status": "Loaded"
        }
        
        return jsonify({
            'success': True, 
            'message': f'Model successfully loaded from {model_path}',
            'model_info': adapter.get_info()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_message = f'Failed to load model: {str(e)}'
        print(f"ERROR: {error_message}")
        current_model_info["status"] = f"Error: {str(e)}"
        return jsonify({'error': error_message}), 500

@app.route('/api/get_model_info', methods=['GET'])
def get_model_info():
    """API endpoint to get information about the currently loaded model"""
    global model_adapter, current_model_info
    
    if model_adapter is not None:
        # Get model info from the adapter
        info = model_adapter.get_info()
    else:
        info = {
            "model_type": "None",
            "model_path": current_model_info.get("path", ""),
            "input_dim": current_model_info.get("input_dim", 2300),
            "output_dim": current_model_info.get("output_dim", 100),
            "loaded": False,
            "status": current_model_info.get("status", "Not loaded")
        }
    
    # Add available model types
    info["available_model_types"] = list(MODEL_ADAPTERS.keys())
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 