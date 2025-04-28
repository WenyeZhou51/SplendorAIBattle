// Define color arrays used by the AI
var colours = ['white', 'blue', 'green', 'red', 'black'];
var all_colours = ['white', 'blue', 'green', 'red', 'black', 'gold'];

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
        
        // Find attainable cards in the market
        let attainable_cards = [];
        for (let tier = 1; tier <= 3; tier++) {
            for (let card of state.cards_in_market[tier]) {
                if (this._isCardAttainable(card, state, player)) {
                    // Calculate efficiency (points per gem)
                    let total_cost = 0;
                    for (let color of colours) {
                        total_cost += card.gems[color] || 0;
                    }
                    
                    // Avoid division by zero
                    if (total_cost === 0) total_cost = 1;
                    
                    attainable_cards.push({
                        card: card,
                        cost: total_cost,
                        efficiency: card.points / total_cost
                    });
                }
            }
        }
        
        // Sort attainable cards by efficiency (highest first)
        attainable_cards.sort((a, b) => {
            if (a.efficiency !== b.efficiency) {
                return b.efficiency - a.efficiency; // Higher efficiency first
            }
            return b.card.points - a.card.points; // Higher points as tiebreaker
        });
        
        // No attainable cards to aim for, get any gem move
        if (attainable_cards.length === 0) {
            return gem_moves[0];
        }
        
        // Target the most efficient card
        let target_card = attainable_cards[0].card;
        
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
    
    _isCardAttainable(card, state, player) {
        // Calculate total gems needed for each color
        let gemsNeeded = {};
        let totalGemsNeeded = 0;
        
        for (let color of colours) {
            // Calculate how many gems we need after discounts
            const required = card.gems[color] || 0;
            const discount = player.card_colours[color] || 0;
            const playerGems = player.gems[color] || 0;
            
            // Net needed = required - discount - player's current gems
            let netNeeded = Math.max(0, required - discount - playerGems);
            
            if (netNeeded > 0) {
                gemsNeeded[color] = netNeeded;
                totalGemsNeeded += netNeeded;
            }
        }
        
        // Check available gems in bank
        let gemsPossibleToGet = 0;
        let goldNeeded = 0;
        
        for (let color of colours) {
            if (gemsNeeded[color] > 0) {
                // How many gems of this color are in the bank
                const bankGems = state.supply_gems[color] || 0;
                
                if (bankGems >= gemsNeeded[color]) {
                    // Bank has enough of this color
                    gemsPossibleToGet += gemsNeeded[color];
                } else {
                    // Bank doesn't have enough, need to use gold
                    gemsPossibleToGet += bankGems;
                    goldNeeded += (gemsNeeded[color] - bankGems);
                }
            }
        }
        
        // Check if player has enough gold to cover the deficit
        const playerGold = player.gems.gold || 0;
        
        // Card is attainable if:
        // 1. We can get enough gems from the bank
        // 2. Player has enough gold to cover any deficit
        return (gemsPossibleToGet + Math.min(goldNeeded, playerGold) >= totalGemsNeeded);
    }
}

class PointRushAI {
    constructor() {
        this.name = "Point Rush";
        this.targetCard = null; // Current card we're targeting
        this.gamePhase = 'early'; // Track game phase: 'early', 'mid', 'late'
        this.turnCount = 0;      // Track number of turns
        this.lastGemColors = []; // Track last collected gem colors to avoid repetition
    }
    
    make_move(state) {
        let moves = state.get_valid_moves();
        let player = state.players[state.current_player_index];
        
        // Update turn count and game phase
        this.turnCount++;
        this._updateGamePhase(player, state);
        
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
            console.log("PointRushAI: Taking winning move!");
            return winning_moves[0];
        }
        
        // Try to buy high-point or highly efficient cards immediately
        const highValueBuyMove = this._findHighValueBuyMove(moves, state, player);
        if (highValueBuyMove) {
            console.log("PointRushAI: Buying high-value card!");
            return highValueBuyMove;
        }
        
        // Look for any decent point card purchase
        const pointBuyMove = this._findPointBuyMove(moves, state);
        if (pointBuyMove) {
            console.log("PointRushAI: Buying point card!");
            return pointBuyMove;
        }
        
        // In late game, try to reserve high-point cards if we can't buy anything good
        if (this.gamePhase === 'late' && player.cards_in_hand.length < 3) {
            const reserveMove = this._findHighPointReserveMove(moves, state);
            if (reserveMove) {
                console.log("PointRushAI: Reserving high-point card!");
                return reserveMove;
            }
        }
        
        // If we can't buy anything, set target card if needed
        if (!this.targetCard || !this._isCardValid(this.targetCard, state)) {
            this.targetCard = this._findBestTargetCard(state, player);
            if (this.targetCard) {
                console.log(`PointRushAI: New target card - ${this._cardDescription(this.targetCard)}`);
            }
        }
        
        // Try to buy any supporting card
        const supportBuyMove = this._findSupportBuyMove(moves, state, player);
        if (supportBuyMove) {
            console.log("PointRushAI: Buying support card!");
            return supportBuyMove;
        }
        
        // Take gems focused on our target card if we have one
        const gemMove = this._findBestGemMove(moves, state, player);
        if (gemMove) {
            // Track the gems we're taking to avoid repetition
            this.lastGemColors = [];
            for (let color in gemMove.gems) {
                if (gemMove.gems[color] > 0) {
                    this.lastGemColors.push(color);
                }
            }
            
            console.log("PointRushAI: Taking gems!");
            return gemMove;
        }
        
        // Reserve tier 3 blind in early-mid game if we have space
        if (this.gamePhase !== 'late' && player.cards_in_hand.length < 3) {
            for (let move of moves) {
                if (move.action === 'reserve' && move.tier === 3 && move.index === -1) {
                    console.log("PointRushAI: Blind reserving tier 3 card!");
                    return move;
                }
            }
        }
        
        // Fallback to any move
        return moves[0];
    }
    
    _updateGamePhase(player, state) {
        const totalScore = player.score;
        const opponentScore = this._getOpponentScore(state, player);
        const scoreDifference = opponentScore - totalScore;
        
        // Transition to late game if anyone is close to winning
        if (totalScore >= 10 || opponentScore >= 10 || this.turnCount > 15) {
            this.gamePhase = 'late';
        } 
        // Transition to mid game earlier if opponent is ahead
        else if (totalScore >= 5 || opponentScore >= 6 || scoreDifference >= 3 || this.turnCount > 8) {
            this.gamePhase = 'mid';
        }
        else {
            this.gamePhase = 'early';
        }
    }
    
    _getOpponentScore(state, player) {
        let maxOpponentScore = 0;
        for (let i = 0; i < state.players.length; i++) {
            if (state.players[i] !== player) {
                maxOpponentScore = Math.max(maxOpponentScore, state.players[i].score);
            }
        }
        return maxOpponentScore;
    }
    
    _findHighValueBuyMove(moves, state, player) {
        let candidates = [];
        
        // Collect high-value buying moves
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                const card = move.card;
                const netCost = this._calculateNetCost(move.gems);
                
                // Calculate a value score
                let valueScore = 0;
                
                // Base score from points, heavily weighted
                valueScore += card.points * 5;
                
                // Early game: prefer cards with at least 2 points
                if (this.gamePhase === 'early' && card.points >= 2) {
                    valueScore += 2;
                }
                
                // Mid/late game: high priority on 3+ point cards
                if (this.gamePhase !== 'early' && card.points >= 3) {
                    valueScore += 5;
                }
                
                // Cost efficiency bonus
                if (netCost > 0) {
                    const efficiency = card.points / netCost;
                    valueScore += efficiency * 3;
                } else {
                    // Free cards are great!
                    valueScore += 5;
                }
                
                // Adjust based on tier in early game
                if (this.gamePhase === 'early' && card.tier === 3) {
                    valueScore -= 2; // Penalty for tier 3 in early game
                }
                
                // Add to candidates if it's a decent value
                if ((card.points >= 2) || (card.points > 0 && netCost <= 3)) {
                    candidates.push({
                        move: move,
                        valueScore: valueScore,
                        points: card.points,
                        netCost: netCost
                    });
                }
            }
        }
        
        if (candidates.length === 0) {
            return null;
        }
        
        // Sort by value score (highest first)
        candidates.sort((a, b) => b.valueScore - a.valueScore);
        
        // Return the highest value move
        return candidates[0].move;
    }
    
    _findPointBuyMove(moves, state) {
        let pointMoves = [];
        
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card && move.card.points > 0) {
                pointMoves.push({
                    move: move,
                    points: move.card.points,
                    cost: this._calculateNetCost(move.gems)
                });
            }
        }
        
        if (pointMoves.length === 0) {
            return null;
        }
        
        // Sort by points (highest first), then by cost (lowest first)
        pointMoves.sort((a, b) => {
            if (a.points !== b.points) {
                return b.points - a.points;
            }
            return a.cost - b.cost;
        });
        
        return pointMoves[0].move;
    }
    
    _findHighPointReserveMove(moves, state) {
        let bestReserveMove = null;
        let bestPoints = 0;
        
        // Try to reserve visible high-point cards
        for (let move of moves) {
            if (move.action === 'reserve' && move.index !== -1) {
                const card = state.cards_in_market[move.tier][move.index];
                if (card.points > bestPoints) {
                    bestPoints = card.points;
                    bestReserveMove = move;
                }
            }
        }
        
        // If no good visible cards, try blind reserve of tier 3
        if (bestPoints < 3) {
            for (let move of moves) {
                if (move.action === 'reserve' && move.tier === 3 && move.index === -1) {
                    return move;
                }
            }
        }
        
        return bestReserveMove;
    }
    
    _isCardValid(card, state) {
        if (!card) return false;
        
        // Check if card is still in market
        const tier = card.tier;
        for (let marketCard of state.cards_in_market[tier]) {
            if (this._isSameCard(card, marketCard)) {
                return true;
            }
        }
        
        return false;
    }
    
    _findBestTargetCard(state, player) {
        // Get all cards in the market
        let allCards = [];
        for (let tier = 1; tier <= 3; tier++) {
            for (let card of state.cards_in_market[tier]) {
                // Skip 0 point cards in mid-late game
                if (this.gamePhase !== 'early' && card.points === 0) {
                    continue;
                }
                
                allCards.push(card);
            }
        }
        
        if (allCards.length === 0) {
            return null;
        }
        
        // Calculate efficiency and attainability for each card
        let candidates = allCards.map(card => {
            // Calculate total gem cost
            let totalCost = 0;
            for (let color of colours) {
                totalCost += card.gems[color] || 0;
            }
            
            // Avoid division by zero
            if (totalCost === 0) totalCost = 1;
            
            // Calculate how attainable the card is
            const attainabilityScore = this._calculateAttainabilityScore(card, state, player);
            
            // Calculate efficiency (points per gem)
            const efficiency = card.points / totalCost;
            
            // Calculate overall score - weight depends on game phase
            let overallScore;
            if (this.gamePhase === 'early') {
                overallScore = (efficiency * 2) + (attainabilityScore * 1);
            } else if (this.gamePhase === 'mid') {
                overallScore = (efficiency * 1.5) + (card.points * 0.5) + (attainabilityScore * 1);
            } else { // late game
                overallScore = (card.points * 1) + (attainabilityScore * 1.5);
            }
            
            return {
                card: card,
                efficiency: efficiency,
                attainability: attainabilityScore,
                overallScore: overallScore,
                points: card.points
            };
        });
        
        // Sort by overall score (highest first)
        candidates.sort((a, b) => b.overallScore - a.overallScore);
        
        // Return the best candidate's card
        return candidates.length > 0 ? candidates[0].card : null;
    }
    
    _calculateAttainabilityScore(card, state, player) {
        // Calculate total gems needed for the card
        let gemsNeeded = {};
        let totalNeeded = 0;
        
        for (let color of colours) {
            const required = card.gems[color] || 0;
            const discount = player.card_colours[color] || 0;
            const playerGems = player.gems[color] || 0;
            
            // Net needed = required - discount - player's current gems
            let netNeeded = Math.max(0, required - discount - playerGems);
            
            if (netNeeded > 0) {
                gemsNeeded[color] = netNeeded;
                totalNeeded += netNeeded;
            }
        }
        
        // Check how many gems we can get from the bank
        let gemsAvailable = 0;
        let goldNeeded = 0;
        
        for (let color of colours) {
            if (gemsNeeded[color] > 0) {
                const bankGems = state.supply_gems[color] || 0;
                
                if (bankGems >= gemsNeeded[color]) {
                    gemsAvailable += gemsNeeded[color];
                } else {
                    gemsAvailable += bankGems;
                    goldNeeded += (gemsNeeded[color] - bankGems);
                }
            }
        }
        
        // Consider player's gold gems
        const playerGold = player.gems.gold || 0;
        const totalGoldAvailable = Math.min(goldNeeded, playerGold);
        
        // Calculate attainability score (0-10)
        // 10 = can buy now, 0 = can't get even with all gems from bank
        if (totalNeeded === 0) {
            return 10; // Can buy now
        }
        
        const totalGemsAttainable = gemsAvailable + totalGoldAvailable;
        const attainabilityRatio = totalGemsAttainable / totalNeeded;
        
        // Convert to 0-10 scale
        return Math.min(10, Math.max(0, attainabilityRatio * 10));
    }
    
    _findSupportBuyMove(moves, state, player) {
        // Buy cards that help with permanent gems or are cheap
        let supportMoves = [];
        
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                const card = move.card;
                const netCost = this._calculateNetCost(move.gems);
                
                // Skip expensive 0-point cards in mid-late game
                if (this.gamePhase !== 'early' && card.points === 0 && netCost > 3) {
                    continue;
                }
                
                // Calculate support score
                let supportScore = 0;
                
                // Low-cost cards are good
                if (netCost <= 2) {
                    supportScore += 3;
                } else if (netCost <= 4) {
                    supportScore += 1;
                }
                
                // Points are good
                supportScore += card.points * 2;
                
                // If we have a target card, permanent gems that help are good
                if (this.targetCard) {
                    const targetCardColor = this._findMajorityColor(this.targetCard.gems);
                    if (targetCardColor && card.colour === targetCardColor) {
                        supportScore += 4;
                    }
                }
                
                supportMoves.push({
                    move: move,
                    supportScore: supportScore,
                    netCost: netCost,
                    points: card.points
                });
            }
        }
        
        if (supportMoves.length === 0) {
            return null;
        }
        
        // Sort by support score (highest first)
        supportMoves.sort((a, b) => b.supportScore - a.supportScore);
        
        return supportMoves[0].move;
    }
    
    _findBestGemMove(moves, state, player) {
        // Get all gem-taking moves
        let gemMoves = moves.filter(move => move.action === 'gems');
        if (gemMoves.length === 0) {
            return null;
        }
        
        // Calculate what gems we need for attainable cards
        let targetNeededGems = {};
        if (this.targetCard) {
            // Calculate needed gems for target card
            for (let color of colours) {
                const required = this.targetCard.gems[color] || 0;
                const discount = player.card_colours[color] || 0;
                const playerGems = player.gems[color] || 0;
                
                // Net needed = required - discount - player's current gems
                let netNeeded = Math.max(0, required - discount - playerGems);
                
                if (netNeeded > 0) {
                    targetNeededGems[color] = netNeeded;
                }
            }
        }
        
        // Find short-term attainable cards
        const attainableCards = this._findPotentialCards(state, player);
        
        // Score each gem move
        let bestMove = null;
        let bestScore = -1;
        
        for (let move of moves) {
            if (move.action !== 'gems') continue;
            
            let score = 0;
            
            // Score based on target card needs
            if (this.targetCard) {
                for (let color in move.gems) {
                    if (move.gems[color] > 0 && targetNeededGems[color] > 0) {
                        score += 2 * move.gems[color];
                    }
                }
            }
            
            // Score based on attainable cards
            for (let attainableCard of attainableCards) {
                for (let color in move.gems) {
                    if (move.gems[color] > 0 && attainableCard.neededGems[color] > 0) {
                        score += 1 * move.gems[color];
                        
                        // Extra bonus for high-point cards
                        score += 0.2 * attainableCard.card.points * move.gems[color];
                    }
                }
            }
            
            // Base score for each gem (taking something is better than nothing)
            for (let color in move.gems) {
                score += 0.1 * move.gems[color];
            }
            
            // Bonus for taking different gems than last time (avoid repetition)
            let diversityBonus = true;
            for (let color in move.gems) {
                if (move.gems[color] > 0 && this.lastGemColors.includes(color)) {
                    diversityBonus = false;
                    break;
                }
            }
            if (diversityBonus && this.lastGemColors.length > 0) {
                score += 1;
            }
            
            // Penalty for taking too many of one color
            for (let color in move.gems) {
                if (move.gems[color] > 0 && player.gems[color] >= 4) {
                    score -= 1;
                }
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        
        return bestMove;
    }
    
    _findPotentialCards(state, player) {
        let potentialCards = [];
        
        // Check each card in the market
        for (let tier = 1; tier <= 3; tier++) {
            for (let card of state.cards_in_market[tier]) {
                // Skip 0 point cards in late game
                if (this.gamePhase === 'late' && card.points === 0) {
                    continue;
                }
                
                // Calculate what gems we still need for this card
                let neededGems = {};
                let totalNeeded = 0;
                
                for (let color of colours) {
                    const required = card.gems[color] || 0;
                    const discount = player.card_colours[color] || 0;
                    const playerGems = player.gems[color] || 0;
                    
                    // Net needed = required - discount - player's current gems
                    let netNeeded = Math.max(0, required - discount - playerGems);
                    
                    if (netNeeded > 0) {
                        neededGems[color] = netNeeded;
                        totalNeeded += netNeeded;
                    }
                }
                
                // If we need 4 or fewer gems total, consider it attainable soon
                if (totalNeeded <= 4) {
                    potentialCards.push({
                        card: card,
                        neededGems: neededGems,
                        totalNeeded: totalNeeded
                    });
                }
            }
        }
        
        // Sort by total gems needed (fewer is better)
        potentialCards.sort((a, b) => {
            // First sort by gems needed
            if (a.totalNeeded !== b.totalNeeded) {
                return a.totalNeeded - b.totalNeeded;
            }
            // Then by points (higher is better)
            return b.card.points - a.card.points;
        });
        
        // Return top 3 potential cards
        return potentialCards.slice(0, 3);
    }
    
    _findMajorityColor(gems) {
        let maxCount = 0;
        let majorityColor = null;
        
        for (let color of colours) {
            const count = gems[color] || 0;
            if (count > maxCount) {
                maxCount = count;
                majorityColor = color;
            }
        }
        
        return majorityColor;
    }
    
    _calculateNetCost(gems) {
        let total = 0;
        for (let color in gems) {
            total += gems[color] || 0;
        }
        return total;
    }
    
    _totalGemCost(gems) {
        let total = 0;
        for (let color in gems) {
            total += gems[color];
        }
        return total;
    }
    
    _isSameCard(card1, card2) {
        if (!card1 || !card2) return false;
        
        // Compare tier, color, points and gem costs
        if (card1.tier !== card2.tier || 
            card1.colour !== card2.colour || 
            card1.points !== card2.points) {
            return false;
        }
        
        // Compare gem costs
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
}

class ColorBot {
    constructor() {
        this.name = "Color Bot";
        this.lockedColors = null; // Will store the two colors we're focusing on
        this.targetCards = null;  // Will store the two most efficient cards we're targeting
        this.gamePhase = 'early'; // Track game phase: 'early', 'mid', 'late'
        this.turnCount = 0;       // Track number of turns for phase transitions
    }
    
    make_move(state) {
        let moves = state.get_valid_moves();
        let player = state.players[state.current_player_index];
        
        // Update turn count and game phase
        this.turnCount++;
        this._updateGamePhase(player);
        
        // Check for winning moves first
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
        
        // If we haven't locked in our colors yet, do that first
        if (!this.lockedColors) {
            this._lockColors(state);
        }
        
        // Log our locked colors for debugging
        if (this.lockedColors) {
            console.log(`ColorBot locked colors: ${this.lockedColors[0]}, ${this.lockedColors[1]}`);
            console.log(`ColorBot target cards: ${this._cardDescription(this.targetCards[0])}, ${this._cardDescription(this.targetCards[1])}`);
            console.log(`Game phase: ${this.gamePhase}`);
        }
        
        // First priority: Try to buy any high-value card (points ≥ 3 or tier 3 card)
        // This makes the bot more opportunistic
        const highValueBuyMove = this._findHighValueBuyMove(moves, state);
        if (highValueBuyMove) {
            return highValueBuyMove;
        }
        
        // Try to buy a target card if possible
        if (this.targetCards) {
            const targetBuyMove = this._findTargetCardBuyMove(moves, state);
            if (targetBuyMove) {
                return targetBuyMove;
            }
        }
        
        // Try to buy a card that gives a permanent gem in our locked colors
        // or a card that costs mainly one of our locked colors and has points
        // More flexible approach with varied priorities by game phase
        const strategicBuyMove = this._findStrategicBuyMove(moves, state);
        if (strategicBuyMove) {
            return strategicBuyMove;
        }
        
        // Reserve one of our target cards if we have space, it's available, and in mid-game
        if (this.targetCards && this.gamePhase !== 'early' && player.cards_in_hand.length < 3) {
            const reserveMove = this._findTargetCardReserveMove(moves, state);
            if (reserveMove) {
                return reserveMove;
            }
        }
        
        // Take gems focused on our locked colors with smarter gem selection
        const gemMove = this._findBestGemMove(moves, state, player);
        if (gemMove) {
            return gemMove;
        }
        
        // Reserve any high-point card if we have space and are in late game
        if (this.gamePhase === 'late' && player.cards_in_hand.length < 3) {
            const reserveHighPointMove = this._findHighPointCardReserveMove(moves, state);
            if (reserveHighPointMove) {
                return reserveHighPointMove;
            }
        }
        
        // Fallback to any move
        return moves[0];
    }
    
    _updateGamePhase(player) {
        const totalScore = player.score;
        const totalCards = player.cards_played.length;
        
        if (totalScore >= 9 || this.turnCount > 15) {
            this.gamePhase = 'late';
        } else if (totalScore >= 4 || totalCards >= 4 || this.turnCount > 8) {
            this.gamePhase = 'mid';
        } else {
            this.gamePhase = 'early';
        }
    }
    
    _lockColors(state) {
        const player = state.players[state.current_player_index];
        
        // Find the two most efficient cards in the market
        let cardEfficiencies = [];
        
        // Check all cards in the market
        for (let tier = 1; tier <= 3; tier++) {
            for (let card of state.cards_in_market[tier]) {
                // Calculate total gem cost
                let totalCost = 0;
                for (let color of colours) {
                    totalCost += card.gems[color] || 0;
                }
                
                // Skip cards with no cost to avoid division by zero
                if (totalCost === 0) continue;
                
                // Calculate points per gem ratio
                const efficiency = card.points / totalCost;
                
                // Find the majority color in the card's cost
                let majorityColor = this._findMajorityColor(card.gems);
                
                cardEfficiencies.push({
                    card: card,
                    efficiency: efficiency,
                    majorityColor: majorityColor
                });
            }
        }
        
        // Sort by efficiency (highest first)
        cardEfficiencies.sort((a, b) => b.efficiency - a.efficiency);
        
        // If we don't have at least two cards, use default colors
        if (cardEfficiencies.length < 2) {
            this.lockedColors = ['red', 'blue']; // Default to red and blue
            this.targetCards = [null, null];
            return;
        }
        
        // Take the two most efficient cards
        const topCard = cardEfficiencies[0];
        const secondCard = cardEfficiencies[1];
        
        // Also consider the colors that appear most frequently in the market
        const colorCounts = this._analyzeMarketColors(state);
        
        // Determine the two colors to lock in
        let color1 = topCard.majorityColor;
        let color2;
        
        if (secondCard.majorityColor !== color1) {
            color2 = secondCard.majorityColor;
        } else {
            // If both top cards have the same majority color,
            // choose the most frequent color in the market as the second color
            let availableColors = [...colours];
            availableColors.splice(availableColors.indexOf(color1), 1);
            
            // Sort by frequency in the market
            availableColors.sort((a, b) => colorCounts[b] - colorCounts[a]);
            color2 = availableColors[0];
        }
        
        this.lockedColors = [color1, color2];
        this.targetCards = [topCard.card, secondCard.card];
    }
    
    _analyzeMarketColors(state) {
        // Count how many times each color appears as a primary cost in the market
        const colorCounts = {
            'white': 0,
            'blue': 0,
            'green': 0,
            'red': 0,
            'black': 0
        };
        
        for (let tier = 1; tier <= 3; tier++) {
            for (let card of state.cards_in_market[tier]) {
                const majorityColor = this._findMajorityColor(card.gems);
                if (majorityColor) {
                    colorCounts[majorityColor]++;
                }
            }
        }
        
        return colorCounts;
    }
    
    _findMajorityColor(gems) {
        let maxCount = 0;
        let majorityColor = null;
        
        for (let color of colours) {
            const count = gems[color] || 0;
            if (count > maxCount) {
                maxCount = count;
                majorityColor = color;
            }
        }
        
        return majorityColor;
    }
    
    _findHighValueBuyMove(moves, state) {
        let highValueMoves = [];
        
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                // Consider any card with 3+ points or tier 3 as high value
                if (move.card.points >= 3 || move.card.tier === 3) {
                    highValueMoves.push({
                        move: move,
                        points: move.card.points,
                        netCost: this._calculateNetGemCost(move.gems)
                    });
                }
            }
        }
        
        if (highValueMoves.length === 0) {
            return null;
        }
        
        // Sort by points (highest first), then by net cost (lowest first)
        highValueMoves.sort((a, b) => {
            if (a.points !== b.points) {
                return b.points - a.points;
            }
            return a.netCost - b.netCost;
        });
        
        return highValueMoves[0].move;
    }
    
    _findTargetCardBuyMove(moves, state) {
        if (!this.targetCards) return null;
        
        // Look for moves that buy one of our target cards
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                for (let targetCard of this.targetCards) {
                    if (targetCard && this._isSameCard(move.card, targetCard)) {
                        return move;
                    }
                }
            }
        }
        
        return null;
    }
    
    _findTargetCardReserveMove(moves, state) {
        if (!this.targetCards) return null;
        
        // Look for moves that reserve one of our target cards
        for (let targetCard of this.targetCards) {
            if (!targetCard) continue;
            
            // Find the card's position in the market
            const tier = targetCard.tier;
            for (let index = 0; index < state.cards_in_market[tier].length; index++) {
                const marketCard = state.cards_in_market[tier][index];
                
                if (this._isSameCard(marketCard, targetCard)) {
                    // Look for a reserve move for this card
                    for (let move of moves) {
                        if (move.action === 'reserve' && 
                            move.tier === tier && 
                            move.index === index) {
                            return move;
                        }
                    }
                }
            }
        }
        
        return null;
    }
    
    _findHighPointCardReserveMove(moves, state) {
        let bestReserveMove = null;
        let bestPoints = 0;
        
        for (let move of moves) {
            if (move.action === 'reserve' && move.tier > 1) { // Focus on tier 2-3 cards
                // If we can see the card
                if (move.index !== -1) {
                    const card = state.cards_in_market[move.tier][move.index];
                    if (card.points > bestPoints) {
                        bestPoints = card.points;
                        bestReserveMove = move;
                    }
                } else if (move.tier === 3 && !bestReserveMove) {
                    // Blind reserve of tier 3 if nothing better
                    bestReserveMove = move;
                }
            }
        }
        
        return bestReserveMove;
    }
    
    _findStrategicBuyMove(moves, state) {
        if (!this.lockedColors) return null;
        
        const player = state.players[state.current_player_index];
        let candidateMoves = [];
        
        // Look at each buying move
        for (let move of moves) {
            if ((move.action === 'buy_available' || move.action === 'buy_reserved') && move.card) {
                const card = move.card;
                const netCost = this._calculateNetGemCost(move.gems);
                
                // Calculate a strategic score for each card
                let strategicScore = 0;
                
                // Base score from card's points
                strategicScore += card.points * 2;
                
                // Permanent gem color is one of our locked colors
                if (this.lockedColors.includes(card.colour)) {
                    strategicScore += 3;
                    
                    // Extra bonus if we're in early game
                    if (this.gamePhase === 'early') {
                        strategicScore += 2;
                    }
                }
                
                // Card's majority cost is one of our locked colors
                const majorityColor = this._findMajorityColor(card.gems);
                if (majorityColor && this.lockedColors.includes(majorityColor)) {
                    strategicScore += 1;
                    
                    // Extra bonus for point cards
                    if (card.points > 0) {
                        strategicScore += 1;
                    }
                }
                
                // Bonus for tier 1 in early game
                if (this.gamePhase === 'early' && card.tier === 1) {
                    strategicScore += 1;
                }
                
                // Penalty for expensive cards in early game
                if (this.gamePhase === 'early' && netCost > 5) {
                    strategicScore -= 2;
                }
                
                // Add to candidates if it has any strategic value
                if (strategicScore > 0) {
                    candidateMoves.push({
                        move: move,
                        strategicScore: strategicScore,
                        netCost: netCost,
                        points: card.points,
                        tier: card.tier
                    });
                }
            }
        }
        
        if (candidateMoves.length === 0) {
            return null;
        }
        
        // Sort by strategic score (higher is better)
        // Break ties with net cost (lower is better), then points (higher is better)
        candidateMoves.sort((a, b) => {
            if (a.strategicScore !== b.strategicScore) {
                return b.strategicScore - a.strategicScore;
            }
            if (a.netCost !== b.netCost) {
                return a.netCost - b.netCost;
            }
            return b.points - a.points;
        });
        
        return candidateMoves[0].move;
    }
    
    _calculateNetGemCost(moveGems) {
        let total = 0;
        for (let color in moveGems) {
            total += moveGems[color];
        }
        return total;
    }
    
    _findBestGemMove(moves, state, player) {
        if (!this.lockedColors) return null;
        
        // Get all gem-taking moves
        let gemMoves = moves.filter(move => move.action === 'gems');
        if (gemMoves.length === 0) {
            return null;
        }
        
        // Calculate what cards we could potentially buy soon
        const potentialCards = this._findPotentialCards(state, player);
        
        // Score each gem move based on how well it helps with our color strategy
        let bestMove = null;
        let bestScore = -1;
        
        for (let move of gemMoves) {
            let score = 0;
            
            // Base score from the gems themselves
            for (let color in move.gems) {
                if (move.gems[color] > 0) {
                    // Core colors get a bonus
                    if (this.lockedColors.includes(color)) {
                        score += 2 * move.gems[color];
                    } else {
                        score += 0.5 * move.gems[color];
                    }
                    
                    // Bonus for colors needed by potential cards
                    for (let potentialCard of potentialCards) {
                        if (potentialCard.neededGems[color] > 0) {
                            score += 1 * move.gems[color];
                            
                            // Extra bonus if the card gives a permanent gem in our locked colors
                            if (this.lockedColors.includes(potentialCard.card.colour)) {
                                score += 0.5 * move.gems[color];
                            }
                        }
                    }
                }
            }
            
            // Special case: taking 2 of a locked color
            for (let color of this.lockedColors) {
                if (move.gems[color] === 2) {
                    score += 2; // Bonus for double gems in our focus colors
                }
            }
            
            // Special case: taking 3 different gems
            let totalDifferentGems = 0;
            for (let color in move.gems) {
                if (move.gems[color] === 1) {
                    totalDifferentGems++;
                }
            }
            if (totalDifferentGems === 3) {
                score += 1.5; // Bonus for diversity when taking 3 different gems
            }
            
            // Account for player's current gem count to avoid waste
            for (let color in move.gems) {
                if (move.gems[color] > 0) {
                    // Penalty if player already has many of this gem (5+)
                    if (player.gems[color] >= 5) {
                        score -= 1.5 * move.gems[color];
                    }
                    // Slight penalty if player already has some of this gem (3-4)
                    else if (player.gems[color] >= 3) {
                        score -= 0.5 * move.gems[color];
                    }
                }
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        
        return bestMove;
    }
    
    _findPotentialCards(state, player) {
        let potentialCards = [];
        
        // Check each card in the market
        for (let tier = 1; tier <= 3; tier++) {
            for (let card of state.cards_in_market[tier]) {
                // Calculate what gems we still need for this card
                let neededGems = {};
                let totalNeeded = 0;
                
                for (let color of colours) {
                    const required = card.gems[color] || 0;
                    const discount = player.card_colours[color] || 0;
                    const playerGems = player.gems[color] || 0;
                    
                    // Net needed = required - discount - player's current gems
                    let netNeeded = Math.max(0, required - discount - playerGems);
                    
                    if (netNeeded > 0) {
                        neededGems[color] = netNeeded;
                        totalNeeded += netNeeded;
                    }
                }
                
                // If we need 5 or fewer gems total, consider it a potential card
                if (totalNeeded <= 5) {
                    potentialCards.push({
                        card: card,
                        neededGems: neededGems,
                        totalNeeded: totalNeeded
                    });
                }
            }
        }
        
        // Sort by total gems needed (fewer is better)
        potentialCards.sort((a, b) => a.totalNeeded - b.totalNeeded);
        
        // Return the top 3 potential cards
        return potentialCards.slice(0, 3);
    }
    
    _isSameCard(card1, card2) {
        if (!card1 || !card2) return false;
        
        // Compare tier, color, points and gem costs
        if (card1.tier !== card2.tier || 
            card1.colour !== card2.colour || 
            card1.points !== card2.points) {
            return false;
        }
        
        // Compare gem costs
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
}

// Create bot instances
ai_neural = new NeuralNetAI('', state_vector_v02);
ai_random = new PointRushAI(); // Replace RandomAI with PointRushAI for backward compatibility
ai_greedy = new GreedyAI();
ai_aggro = new PointRushAI(); // Replace AggroAI with PointRushAI for backward compatibility
ai_point_rush = new PointRushAI();
ai_color = new ColorBot();

// Default AI
ai = ai_neural;

