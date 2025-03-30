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

// Create bot instances
ai_neural = new NeuralNetAI('', state_vector_v02);
ai_random = new RandomAI();
ai_greedy = new GreedyAI();

// Default AI
ai = ai_neural;

class AggroAI {
    constructor() {
        this.name = "Aggro";
        this.targetCard = null;
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
            // Reset target card since we're about to win
            this.targetCard = null;
            return winning_moves[0];
        }
        
        // Find buyable target-type cards (always reevaluate)
        const buyableTargetMoves = this._findTargetCardPurchaseMoves(moves);
        if (buyableTargetMoves.length > 0) {
            // Reset target card since we're buying one of our targets
            this.targetCard = null;
            return buyableTargetMoves[0];
        }
        
        // Find or update our target card if we don't have one or it's no longer available
        if (!this.targetCard || !this._isCardStillAvailable(this.targetCard, state)) {
            this.targetCard = this._findBestTargetCard(state);
        }
        
        // If we have a target card and it's affordable, buy it
        if (this.targetCard) {
            const buyMove = this._getBuyMoveForCard(this.targetCard, moves);
            if (buyMove) {
                // Reset target card since we're buying it
                this.targetCard = null;
                return buyMove;
            }
            
            // If we can't buy it yet, collect gems for it
            const gemMove = this._findGemMoveForTargetCard(this.targetCard, moves, state);
            if (gemMove) {
                return gemMove;
            }
        }
        
        // Fallback: Buy any card with points (prioritize higher tier)
        const pointCardMoves = this._findAnyPointCardMoves(moves);
        if (pointCardMoves.length > 0) {
            return pointCardMoves[0];
        }
        
        // Last resort: Take random gems
        const gemMoves = moves.filter(move => move.action === 'gems');
        if (gemMoves.length > 0) {
            return gemMoves[0];
        }
        
        // Absolute fallback
        return moves[0];
    }
    
    _isCardStillAvailable(card, state) {
        if (!card) return false;
        
        // Check if card is in the market
        for (let tier = 1; tier <= 3; tier++) {
            for (let marketCard of state.cards_in_market[tier]) {
                if (this._isSameCard(card, marketCard)) {
                    return true;
                }
            }
        }
        
        // Check if it's reserved by the player
        const player = state.players[state.current_player_index];
        for (let reservedCard of player.cards_in_hand) {
            if (this._isSameCard(card, reservedCard)) {
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
    
    _findTargetCardPurchaseMoves(moves) {
        let targetMoves = [];
        
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                // Check that the card has points
                if (move.card.points <= 0) continue;
                
                if (
                    // Level 3 with 7 of a single gem
                    (move.card.tier === 3 && this._hasSingleColor7Gems(move.card)) ||
                    
                    // Level 2 with 5 of a single gem
                    (move.card.tier === 2 && this._hasSingleColor5Gems(move.card)) ||
                    
                    // Level 2 with 6 of a single gem
                    (move.card.tier === 2 && this._hasSingleColor6Gems(move.card)) ||
                    
                    // Level 2 with 2,4,1 distribution
                    (move.card.tier === 2 && this._has241GemDistribution(move.card))
                ) {
                    targetMoves.push(move);
                }
            }
        }
        
        // Sort by points (highest first)
        targetMoves.sort((a, b) => b.card.points - a.card.points);
        
        return targetMoves;
    }
    
    _findBestTargetCard(state) {
        let targetCards = [];
        const player = state.players[state.current_player_index];
        
        // Check market for target cards
        for (let tier of [3, 2]) { // Check tier 3 first, then tier 2
            for (let card of state.cards_in_market[tier]) {
                // Only consider cards with points
                if (card.points <= 0) continue;
                
                if (
                    // Level 3 with 7 of a single gem
                    (tier === 3 && this._hasSingleColor7Gems(card)) ||
                    
                    // Level 2 with 5 of a single gem
                    (tier === 2 && this._hasSingleColor5Gems(card)) ||
                    
                    // Level 2 with 6 of a single gem
                    (tier === 2 && this._hasSingleColor6Gems(card)) ||
                    
                    // Level 2 with 2,4,1 distribution
                    (tier === 2 && this._has241GemDistribution(card))
                ) {
                    targetCards.push(card);
                }
            }
            
            // If we found any target cards in this tier, don't check lower tiers
            if (targetCards.length > 0) break;
        }
        
        // Also check player's reserved cards
        for (let card of player.cards_in_hand) {
            // Only consider cards with points
            if (card.points <= 0) continue;
            
            if (
                // Level 3 with 7 of a single gem
                (card.tier === 3 && this._hasSingleColor7Gems(card)) ||
                
                // Level 2 with 5 of a single gem
                (card.tier === 2 && this._hasSingleColor5Gems(card)) ||
                
                // Level 2 with 6 of a single gem
                (card.tier === 2 && this._hasSingleColor6Gems(card)) ||
                
                // Level 2 with 2,4,1 distribution
                (card.tier === 2 && this._has241GemDistribution(card))
            ) {
                targetCards.push(card);
            }
        }
        
        // If no target cards found, check for any point cards in tier 2 or 1
        if (targetCards.length === 0) {
            for (let tier of [2, 1]) {
                for (let card of state.cards_in_market[tier]) {
                    if (card.points > 0) {
                        targetCards.push(card);
                    }
                }
                
                // If we found any point cards in this tier, don't check lower tiers
                if (targetCards.length > 0) break;
            }
        }
        
        // Sort by tier then by points (prioritize higher tier and higher points)
        targetCards.sort((a, b) => {
            if (a.tier !== b.tier) {
                return b.tier - a.tier;
            }
            return b.points - a.points;
        });
        
        return targetCards.length > 0 ? targetCards[0] : null;
    }
    
    _getBuyMoveForCard(card, moves) {
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && 
                move.card && this._isSameCard(card, move.card)) {
                return move;
            }
        }
        return null;
    }
    
    _findGemMoveForTargetCard(card, moves, state) {
        const player = state.players[state.current_player_index];
        const gemMoves = moves.filter(move => move.action === 'gems');
        
        if (gemMoves.length === 0 || !card) return null;
        
        // Calculate what gems we need for the target card
        let neededGems = {};
        for (let color of colours) {
            const required = card.gems[color] || 0;
            const discount = player.card_colours[color] || 0;
            const playerGems = player.gems[color] || 0;
            
            const stillNeeded = Math.max(0, required - discount - playerGems);
            if (stillNeeded > 0) {
                neededGems[color] = stillNeeded;
            }
        }
        
        // Score each gem move based on how much it helps with the target card
        let bestMove = null;
        let bestScore = -1;
        
        for (let move of gemMoves) {
            let score = 0;
            for (let color in move.gems) {
                if (move.gems[color] > 0 && neededGems[color] > 0) {
                    score += Math.min(move.gems[color], neededGems[color]);
                }
            }
            
            // Bonus for taking 2 of the same color when we need 2+ of that color
            for (let color in move.gems) {
                if (move.gems[color] === 2 && neededGems[color] >= 2) {
                    score += 0.5;
                }
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        
        return bestMove || gemMoves[0];
    }
    
    _findAnyPointCardMoves(moves) {
        const pointCardMoves = [];
        
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && 
                move.card && move.card.points > 0) {
                pointCardMoves.push(move);
            }
        }
        
        // Sort by tier then points
        pointCardMoves.sort((a, b) => {
            if (a.card.tier !== b.card.tier) {
                return b.card.tier - a.card.tier;
            }
            return b.card.points - a.card.points;
        });
        
        return pointCardMoves;
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

