// Define color arrays used by the AI
var colours = ['white', 'blue', 'green', 'red', 'black'];
var all_colours = ['white', 'blue', 'green', 'red', 'black', 'gold'];

class RandomAI {
    make_move(state) {
        var choice;
        let moves = state.get_valid_moves();
        let player = state.players[state.current_player_index];

        if (player.total_num_gems() <= 8) {
            let gems_moves = [];
            for (let move of moves) {
                if (move['action'] === 'gems') {
                    gems_moves.push(move);
                }
            }
            if (gems_moves.length > 0 && state.total_num_gems_available() >= 3) {
                choice = math.pickRandom(gems_moves);
                return choice;
            }
        }

        let buying_moves = [];
        for (let move of moves) {
            if (move['action'] === 'buy_available' || move['action'] === 'buy_reserved') {
                buying_moves.push(move);
            }
        }
        if (buying_moves.length > 0) {
            choice = math.pickRandom(buying_moves);
            return choice
        }

        let reserving_moves = [];
        for (let move of moves) {
            if (move['action'] === 'reserve') {
                reserving_moves.push(move);
            }
        }
        if (reserving_moves.length > 0) {
            choice = math.pickRandom(reserving_moves);
            return choice;
        }

        return math.pickRandom(moves);
    }
}

function relu(arr) {
    for (let row_index = 0; row_index < arr.size()[0]; row_index++) {
        let row = arr._data[row_index];
        for (let i = 0; i < row.length; i++) {
            let value = row[i];
            if (value < 0) {
                row[i] = 0;
            }
        }
    }
    return arr;
}

function softmax(arr, prob_factor = 1) {
    let exp_arr = [];
    for (let row of arr) {
        let sum = 0;
        let exp_row = [];
        for (let value of row) {
            let exp_value = Math.exp(prob_factor * value);
            sum += exp_value;
            exp_row.push(exp_value);
        }
        for (let i = 0; i < row.length; i++) {
            exp_row[i] /= sum;
        }
        exp_arr.push(exp_row);
    }
    return exp_arr;
}

class NeuralNetAI {
    constructor(identifier='', prob_factor=100, state_vector) {
        this.weight_1 = math.matrix(weights['weight_1' + identifier]);
        this.weight_2 = math.matrix(weights['weight_2' + identifier]);
        this.bias_1 = math.matrix(weights['bias_1' + identifier]);
        this.bias_2 = math.matrix(weights['bias_2' + identifier]);

        this.prob_factor = prob_factor;
    }

    make_move(state) {
        let moves = state.get_valid_moves();
        let player = state.players[state.current_player_index];
        let current_player_index = state.current_player_index;

        let input_vector = [];
        let scores = [];
        for (let move of moves) {
            let cur_state = state.copy();
            cur_state.make_move(move);
            input_vector.push(cur_state.get_state_vector(current_player_index));
            scores.push(cur_state.players[current_player_index].score);
        }

        // If we can get to 15 points, do so
        let best_score = 0;
        let best_index = 0;
        for (let i = 0; i < moves.length; i++) {
            let score = scores[i];
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }
        if (best_score >= 15) {
            return moves[best_index];
        }

        // input_vector = vectors;

        // console.log('Using fake input vector');
        // input_vector = vectors;

        // console.log('pre', math.matrix(input_vector), this.weight_1);

        let hidden_output_1 = math.multiply(
            math.matrix(input_vector), this.weight_1);
        // console.log('first multiply', hidden_output_1);
        let bias_1_arr = [];
        for (let i = 0; i < hidden_output_1.size()[0]; i++) {
            bias_1_arr.push(this.bias_1._data);
        }
        hidden_output_1 = math.add(hidden_output_1, bias_1_arr);
    
        // console.log('before relu', hidden_output_1);
        hidden_output_1 = relu(hidden_output_1);
        // console.log('after relu', hidden_output_1);

        hidden_output_1 = math.matrix(hidden_output_1);

        let output = math.multiply(
            hidden_output_1, this.weight_2);
        // console.log('intermediate 2', output);
        let bias_2_arr = [];
        for (let i = 0; i < hidden_output_1.size()[0]; i++) {
            bias_2_arr.push(this.bias_2._data);
        }
        output = math.add(output, bias_2_arr);
        output = softmax(output._data);

        let probabilities_input = [];
        for (let row of output) {
            probabilities_input.push(row[0]);
        }
        let probabilities = softmax([probabilities_input], 100)[0];
        console.log('probabilities are', probabilities)

        let number = math.random();
        let move = null;
        let cumulative_probability = 0;
        for (let i = 0; i < probabilities.length; i++) {
            cumulative_probability += probabilities[i];
            if (cumulative_probability > number) {
                move = moves[i];
                break;
            }
        }
        if (move === null) {
            console.log('Failed to choose a move using softmax probabilities')
            move = moves[moves.length - 1];
        }

        return move;

    }
}

// ai_v01 = new NeuralNetAI('', state_vector_v01);
// ai = ai_v01;

ai_v02 = new NeuralNetAI('', state_vector_v02);
ai = ai_v02;

class GreedyAI {
    constructor() {
        this.name = "Greedy";
    }
    
    make_move(state) {
        let moves = state.get_valid_moves();
        let player = state.players[state.current_player_index];
        
        // If we can win, always choose that move
        let winning_moves = [];
        for (let move of moves) {
            let new_state = state.copy();
            new_state.make_move(move);
            if (new_state.players[state.current_player_index].score >= 15) {
                winning_moves.push(move);
            }
        }
        
        if (winning_moves.length > 0) {
            return winning_moves[0];
        }
        
        // Look for card purchase moves
        let buying_moves = [];
        for (let move of moves) {
            if (move.action === 'buy_available' || move.action === 'buy_reserved') {
                buying_moves.push(move);
            }
        }
        
        // Sort buying moves by cost (lowest first), then by points (highest) for tiebreakers
        if (buying_moves.length > 0) {
            buying_moves.sort((a, b) => {
                // First prioritize by total gem cost
                const aGemCost = this._totalGemCost(a.gems);
                const bGemCost = this._totalGemCost(b.gems);
                
                if (aGemCost !== bGemCost) {
                    return aGemCost - bGemCost; // Lower gem cost is better
                }
                
                // If costs are equal, prioritize by points
                const aPoints = a.card ? a.card.points : 0;
                const bPoints = b.card ? b.card.points : 0;
                return bPoints - aPoints; // Higher points as tiebreaker
            });
            
            return buying_moves[0];
        }
        
        // Take gems to prepare for future purchases
        return this._findBestGemMove(moves, state, player);
    }
    
    _totalGemCost(gems) {
        let total = 0;
        for (let color in gems) {
            total += gems[color];
        }
        return total;
    }
    
    _findBestGemMove(moves, state, player) {
        // Get all gem-taking moves
        let gem_moves = moves.filter(move => move.action === 'gems');
        if (gem_moves.length === 0) {
            return moves[0]; // Fallback to any move
        }
        
        // Find the cheapest cards in the market
        let available_cards = [];
        for (let tier = 1; tier <= 3; tier++) { // Start with tier 1 (cheapest)
            for (let card of state.cards_in_market[tier]) {
                // Calculate effective cost after applying discounts
                let effective_cost = 0;
                for (let color of colours) {
                    const required = card.gems[color] || 0;
                    const discount = player.card_colours[color] || 0;
                    effective_cost += Math.max(0, required - discount);
                }
                
                available_cards.push({
                    card: card,
                    cost: effective_cost
                });
            }
        }
        
        // Sort cards by cost (lowest first)
        available_cards.sort((a, b) => a.cost - b.cost);
        
        // No cards to aim for, get any gem move
        if (available_cards.length === 0) {
            return gem_moves[0];
        }
        
        // Target the cheapest card
        let target_card = available_cards[0].card;
        
        // Find which gems we need to get the target card
        let needed_gems = {};
        for (let color of colours) {
            const required = target_card.gems[color] || 0;
            const discount = player.card_colours[color] || 0;
            const gems_owned = player.gems[color] || 0;
            
            let still_needed = Math.max(0, required - discount - gems_owned);
            if (still_needed > 0) {
                needed_gems[color] = still_needed;
            }
        }
        
        // Find gem move that best helps us get those gems
        let best_move = null;
        let best_score = -1;
        
        for (let move of gem_moves) {
            let score = 0;
            for (let color in move.gems) {
                if (move.gems[color] > 0 && needed_gems[color] > 0) {
                    score += move.gems[color];
                }
            }
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        if (best_move) {
            return best_move;
        }
        
        // If no good move found, take two of a color if possible
        for (let move of gem_moves) {
            for (let color in move.gems) {
                if (move.gems[color] === 2) {
                    return move;
                }
            }
        }
        
        // Otherwise, get any gem move
        return gem_moves[0];
    }
}

class AggroAI {
    constructor() {
        this.name = "Aggro";
        this.winConditionCard = null; // Card we're targeting as win condition
    }
    
    make_move(state) {
        let moves = state.get_valid_moves();
        let player = state.players[state.current_player_index];
        
        // First check for winning moves
        let winning_moves = [];
        for (let move of moves) {
            let new_state = state.copy();
            new_state.make_move(move);
            if (new_state.players[state.current_player_index].score >= 15) {
                winning_moves.push(move);
            }
        }
        
        if (winning_moves.length > 0) {
            // Reset win condition since we're about to win
            this.winConditionCard = null;
            return winning_moves[0];
        }
        
        // If we don't have a win condition card yet, or it's no longer valid, find a new one
        if (!this.winConditionCard || !this._isWinConditionValid(this.winConditionCard, state)) {
            // Look for a new win condition card
            const newWinCondition = this._findBestWinCondition(state);
            
            if (newWinCondition) {
                this.winConditionCard = newWinCondition;
                console.log(`AggroAI set new win condition: ${this._cardDescription(this.winConditionCard)}`);
                
                // If we have fewer than 3 reserved cards, reserve this win condition
                const reserveMove = this._getReserveMoveForCard(this.winConditionCard, moves, state);
                if (reserveMove && player.cards_in_hand.length < 3) {
                    return reserveMove;
                }
            }
        }
        
        // If we have a win condition in our reserved cards that we can buy, buy it
        const reservedWinCondition = this._findReservedWinCondition(player);
        if (reservedWinCondition) {
            const buyReservedMove = this._getBuyReservedMove(reservedWinCondition, moves);
            if (buyReservedMove) {
                // We've achieved this win condition, reset to find a new one
                this.winConditionCard = null;
                return buyReservedMove;
            }
        }
        
        // Try to buy permanent gems that help with our win condition
        const helpfulCardMove = this._findHelpfulCardPurchase(moves, state);
        if (helpfulCardMove) {
            return helpfulCardMove;
        }
        
        // If we have fewer than 3 cards and our win condition is not reserved yet, reserve it
        if (player.cards_in_hand.length < 3 && !this._isCardReserved(this.winConditionCard, player)) {
            const reserveMove = this._getReserveMoveForCard(this.winConditionCard, moves, state);
            if (reserveMove) {
                return reserveMove;
            }
        }
        
        // Take gems that help with our win condition
        const gemMove = this._findHelpfulGemMove(moves, state);
        if (gemMove) {
            return gemMove;
        }
        
        // Fallback: Take any gems
        const anyGemMove = moves.filter(move => move.action === 'gems');
        if (anyGemMove.length > 0) {
            return anyGemMove[0];
        }
        
        // Absolute fallback
        return moves[0];
    }
    
    _isWinConditionValid(card, state) {
        if (!card) return false;
        
        // Check if the card is still in the market (if not already reserved)
        const player = state.players[state.current_player_index];
        
        // If it's in our reserved cards, it's valid
        if (this._isCardReserved(card, player)) {
            return true;
        }
        
        // Otherwise check if it's still in the market
        return this._isCardInMarket(card, state);
    }
    
    _isCardReserved(card, player) {
        if (!card) return false;
        
        for (let reservedCard of player.cards_in_hand) {
            if (this._isSameCard(card, reservedCard)) {
                return true;
            }
        }
        
        return false;
    }
    
    _isCardInMarket(card, state) {
        if (!card) return false;
        
        const tier = card.tier;
        for (let marketCard of state.cards_in_market[tier]) {
            if (this._isSameCard(card, marketCard)) {
                return true;
            }
        }
        
        return false;
    }
    
    _isSameCard(card1, card2) {
        if (!card1 || !card2) return false;
        
        // Compare tier, color, points and gem costs
        return card1.tier === card2.tier && 
               card1.colour === card2.colour &&
               card1.points === card2.points &&
               this._haveSameGemCost(card1, card2);
    }
    
    _haveSameGemCost(card1, card2) {
        for (let color of colours) {
            if ((card1.gems[color] || 0) !== (card2.gems[color] || 0)) {
                return false;
            }
        }
        return true;
    }
    
    _cardDescription(card) {
        if (!card) return "no card";
        
        let gemDesc = "";
        for (let color of colours) {
            if (card.gems[color] > 0) {
                gemDesc += `${card.gems[color]} ${color}, `;
            }
        }
        
        return `Tier ${card.tier} ${card.colour} card with ${card.points} points (${gemDesc.slice(0, -2)})`;
    }
    
    _findBestWinCondition(state) {
        const player = state.players[state.current_player_index];
        let candidates = [];
        
        // Check tier 3 cards first
        for (let card of state.cards_in_market[3]) {
            // Check for 7 single gem or 6,3,3 composition
            if (this._hasSingleColor7Gems(card) || this._has633GemDistribution(card)) {
                candidates.push(card);
            }
        }
        
        // If no tier 3 candidates, check tier 2 cards
        if (candidates.length === 0) {
            for (let card of state.cards_in_market[2]) {
                // Check for 5 single gem, 6 single gem, or 2,4,1 composition
                if (this._hasSingleColor5Gems(card) || 
                    this._hasSingleColor6Gems(card) || 
                    this._has241GemDistribution(card)) {
                    candidates.push(card);
                }
            }
        }
        
        // If no candidates found, return null
        if (candidates.length === 0) {
            return null;
        }
        
        // Sort by tier first, then points
        candidates.sort((a, b) => {
            if (a.tier !== b.tier) {
                return b.tier - a.tier; // Higher tier first
            }
            return b.points - a.points; // Higher points second
        });
        
        return candidates[0];
    }
    
    _findReservedWinCondition(player) {
        for (let card of player.cards_in_hand) {
            // Check if this reserved card matches our win condition criteria
            if (card.tier === 3 && (this._hasSingleColor7Gems(card) || this._has633GemDistribution(card))) {
                return card;
            }
            
            if (card.tier === 2 && (this._hasSingleColor5Gems(card) || 
                                    this._hasSingleColor6Gems(card) || 
                                    this._has241GemDistribution(card))) {
                return card;
            }
        }
        
        return null;
    }
    
    _getReserveMoveForCard(card, moves, state) {
        if (!card) return null;
        
        // Find the card's position in the market
        const tier = card.tier;
        for (let index = 0; index < state.cards_in_market[tier].length; index++) {
            const marketCard = state.cards_in_market[tier][index];
            
            if (this._isSameCard(card, marketCard)) {
                // Find a reserve move for this card
                for (let move of moves) {
                    if (move.action === 'reserve' && 
                        move.tier === tier && 
                        move.index === index) {
                        return move;
                    }
                }
            }
        }
        
        return null;
    }
    
    _getBuyReservedMove(card, moves) {
        if (!card) return null;
        
        // Find the reserved card index
        for (let move of moves) {
            if (move.action === 'buy_reserved' && move.card && this._isSameCard(card, move.card)) {
                return move;
            }
        }
        
        return null;
    }
    
    _colorHelpsWithWinCondition(color, winCondition, player) {
        if (!winCondition) return false;
        
        // Check if the win condition requires this color
        return (winCondition.gems[color] || 0) > (player.card_colours[color] || 0);
    }
    
    _findHelpfulCardPurchase(moves, state) {
        if (!this.winConditionCard) return null;
        
        const player = state.players[state.current_player_index];
        let helpfulMoves = [];
        
        // Look for cards that provide permanent gems that help with our win condition
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                // Check if this card's color helps with our win condition
                if (this._colorHelpsWithWinCondition(move.card.colour, this.winConditionCard, player)) {
                    helpfulMoves.push(move);
                }
            }
        }
        
        if (helpfulMoves.length === 0) {
            return null;
        }
        
        // Sort by cost (cheapest first)
        helpfulMoves.sort((a, b) => {
            const aCost = this._calculateGemCost(a.gems);
            const bCost = this._calculateGemCost(b.gems);
            
            if (aCost !== bCost) {
                return aCost - bCost; // Cheapest first
            }
            
            // If same cost, prefer higher points
            return (b.card.points || 0) - (a.card.points || 0);
        });
        
        return helpfulMoves[0];
    }
    
    _calculateGemCost(gems) {
        let total = 0;
        for (let color of all_colours) {
            total += gems[color] || 0;
        }
        return total;
    }
    
    _findHelpfulGemMove(moves, state) {
        if (!this.winConditionCard) return null;
        
        const player = state.players[state.current_player_index];
        const gemMoves = moves.filter(move => move.action === 'gems');
        
        if (gemMoves.length === 0) {
            return null;
        }
        
        // Calculate what gems we need for our win condition
        let neededGems = {};
        let totalNeeded = 0;
        
        for (let color of colours) {
            // Check how many gems of this color we need
            const required = this.winConditionCard.gems[color] || 0;
            const discount = player.card_colours[color] || 0;
            const playerGems = player.gems[color] || 0;
            
            // We need: (requirement - discount - current gems)
            const stillNeeded = Math.max(0, required - discount - playerGems);
            
            if (stillNeeded > 0) {
                neededGems[color] = stillNeeded;
                totalNeeded += stillNeeded;
            }
        }
        
        // Score each gem move by how helpful it is
        let bestMove = null;
        let bestScore = -1;
        
        for (let move of gemMoves) {
            let score = 0;
            
            for (let color in move.gems) {
                if (move.gems[color] > 0 && neededGems[color] > 0) {
                    // Award points based on how much this helps with our needed gems
                    score += Math.min(move.gems[color], neededGems[color]);
                    
                    // Bonus for taking 2 of a color we need a lot of
                    if (move.gems[color] === 2 && neededGems[color] >= 3) {
                        score += 1;
                    }
                }
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        
        // If we found a helpful move, return it
        if (bestMove && bestScore > 0) {
            return bestMove;
        }
        
        // Otherwise, return the first gem move
        return gemMoves[0];
    }
    
    _hasSingleColor7Gems(card) {
        if (!card) return false;
        
        for (let color of colours) {
            if (card.gems[color] === 7) {
                return true;
            }
        }
        return false;
    }
    
    _has633GemDistribution(card) {
        if (!card) return false;
        
        // Count gems by color
        let counts = {};
        let totalGems = 0;
        
        for (let color of colours) {
            const count = card.gems[color] || 0;
            if (count > 0) {
                counts[count] = (counts[count] || 0) + 1;
                totalGems += count;
            }
        }
        
        // Check for 6,3,3 pattern (one color with 6, two colors with 3 each)
        return counts[6] === 1 && counts[3] === 2 && totalGems === 12;
    }
    
    _hasSingleColor6Gems(card) {
        if (!card) return false;
        
        for (let color of colours) {
            if (card.gems[color] === 6) {
                return true;
            }
        }
        return false;
    }
    
    _hasSingleColor5Gems(card) {
        if (!card) return false;
        
        for (let color of colours) {
            if (card.gems[color] === 5) {
                return true;
            }
        }
        return false;
    }
    
    _has241GemDistribution(card) {
        if (!card) return false;
        
        // Count gems by color
        let counts = {};
        let totalGems = 0;
        
        for (let color of colours) {
            const count = card.gems[color] || 0;
            if (count > 0) {
                counts[count] = (counts[count] || 0) + 1;
                totalGems += count;
            }
        }
        
        // Check for 2,4,1 pattern
        return counts[2] === 1 && counts[4] === 1 && counts[1] === 1 && totalGems === 7;
    }
}

// Create bot instances
ai_neural = new NeuralNetAI('', state_vector_v02);
ai_random = new RandomAI();
ai_greedy = new GreedyAI();
ai_aggro = new AggroAI();

// Default AI
ai = ai_neural;

