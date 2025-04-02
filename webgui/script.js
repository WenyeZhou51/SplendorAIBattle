var background_colours = {
    'white': '#ffffff',
    'blue': '#bbbbff',
    'red': '#ffbbbb',
    'green': '#bbffbb',
    'black': '#aaaaaa',
    'gold': '#ffffbb',
};

var border_colours = {
    'white': 'lightgrey',
    'blue': 'blue',
    'red': 'red',
    'green': 'green',
    'black': 'black',
    'gold': 'gold',
};

var a = math.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
var b = math.matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]]);



// shuffle function from https://bost.ocks.org/mike/algorithms/#shuffling
function shuffle(array) {
    var n = array.length, t, i;
    while (n) {
        i = Math.random() * n-- | 0; // 0 ≤ i < n
        t = array[n];
        array[n] = array[i];
        array[i] = t;
    }
    return array;
}


var test_state = new GameState();


Vue.component('gems-table', {
    props: ['gems', 'cards', 'show_card_count'],
    template: `
<table class="gems-table">
  <tr>
    <gems-table-gem-counter v-for="(number, colour) in gems"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-gem-counter>
  </tr>
  <tr v-if="show_card_count">
    <gems-table-card-counter v-for="(number, colour) in cards"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-card-counter>
  </tr>
</table>
`
});

Vue.component('gems-table-gem-counter', {
    props: ['colour', 'number'],
    computed: {
        border_colour: function() {
            return border_colours[this.colour];
        },
        background_colour: function() {
            return background_colours[this.colour];
        }
    },
    template: `
<td class="gems-table-gem-counter"
    v-bind:style="{background: background_colour, borderColor: border_colour}">
  {{ number }}
</td>
`
});

Vue.component('gems-table-card-counter', {
    props: ['colour', 'number'],
    computed: {
        border_colour: function() {
            return border_colours[this.colour];
        },
        background_colour: function() {
            return background_colours[this.colour];
        }
    },
    template: `
<td class="gems-table-card-counter"
    v-bind:style="{background: background_colour, borderColor: border_colour}">
  {{ number }}
</td>
`
});


Vue.component('gems-list', {
    props: {gems: {},
            title: "",
            display_zeros: {default: true}},
    template: `
<div class="gems-list">
    <h3 v-if="title">{{ title }}</h3>
    <ul>
    <gem-counter 
        v-for="(number, colour) in gems"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number"
        style="font-size:2vh;"
        v-if="number > 0 || display_zeros">
    </gem-counter>
    </ul>
</div>`
});

Vue.component('gem-counter', {
    props: ['colour', 'number'],
    computed: {
        border_colour: function() {
            return border_colours[this.colour];
        },
        background_colour: function() {
            return background_colours[this.colour];
        }
    },
    template: `
<li class="gem-counter" 
    v-bind:style="{background: background_colour, borderColor: border_colour}">
  {{ number }}
</li>
`
});

Vue.component('gem-discarder', {
    props: ['player', 'gems_discarded', 'player_gems'],
    methods: {
        discard_gems: function() {
            this.$emit('discard_gems');
        }
    },
    template: `
<div class="gem-discarder">
  <h3>discard gems</h3>
  <gem-discarder-table v-bind:gems_discarded="gems_discarded"
                       v-bind:player_gems="player.gems">
  </gem-discarder-table>
  <button v-on:click="discard_gems()">
    confirm discards
  </button>
</div>
`
});

Vue.component('gem-discarder-table', {
    props: ['player_gems', 'gems_discarded'],
    methods: {
        increment(colour) {
            this.player_gems[colour] += 1;
            this.gems_discarded[colour] -= 1;
        },
        decrement(colour) {
            this.player_gems[colour] -= 1;
            this.gems_discarded[colour] += 1;
        }
    },
    computed: {
        total_player_gems: function() {
            let total_gems = 0;
            for (let colour of all_colours) {
                total_gems += this.player_gems[colour];
            }
            return total_gems;
        },
        can_increment: function() {
            let incrementable = {};
            for (let colour of all_colours) {
                incrementable[colour] = this.gems_discarded[colour] > 0;
            }
            return incrementable;
        },
        can_decrement: function() {
            let decrementable = {};
            let total_gems = this.total_player_gems;
            for (let colour of all_colours) {
                decrementable[colour] = total_gems > 10 && this.player_gems[colour] > 0;
            }
            return decrementable;
        },
        computed: {
            show_button: function() {
                return true;
            }
        }
    },
    template: `
<table class="gem-discarder-table">
  <tr>
    <td>
        current gems
    </td>
    <gems-table-gem-counter v-for="(number, colour) in player_gems"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-gem-counter>
  </tr>
  <tr>
    <td>
    </td>
    <increment-button v-for="(number, colour) in player_gems"
                      v-bind:key="colour"
                      v-bind:enabled="can_increment[colour]"
                      v-bind:show_button="true"
                      v-on:increment="increment($event)"
                      v-bind:colour="colour">
 e  </increment-button>
  </tr>
  <tr>
    <td>
    </td>
    <decrement-button v-for="(number, colour) in player_gems"
                      v-bind:key="colour"
                      v-bind:enabled="can_decrement[colour]"
                      v-bind:show_button="true"
                      v-on:decrement="decrement($event)"
                      v-bind:colour="colour">
    </decrement-button>
  </tr>
  <tr>
    <td>
        discarded gems
    </td>
    <gems-table-gem-counter v-for="(number, colour) in gems_discarded"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-gem-counter>
  </tr>
</table>
`
});

Vue.component('move-maker', {
    props: ['player', 'supply_gems', 'gems', 'player_gems', 'player_cards'],
    methods: {
        take_gems: function() {
            this.$emit('take_gems', this.gems);
        }
    },
    computed: {
        any_gems_selected: function() {
            var any_gems_in_supply = false;
            for (let colour of colours) {
                if (this.supply_gems[colour] > 0) {
                    any_gems_in_supply = true;
                    break;
                }
            }

            if (!any_gems_in_supply) {
                return true;  // no gems in supply, so can 'take gems'
                              // without selecting any
            }

            var any_selected = false;
            for (let colour of colours) {
                if (this.gems[colour] > 0) {
                    any_selected = true;
                    break;
                }
            }
            return any_selected;
        }
    },
    template: `
<div class="move-maker">
  <gem-selector v-bind:supply_gems="supply_gems"
                v-bind:player_gems="player_gems"
                v-bind:player_cards="player_cards"
                v-bind:gems="gems">
  </gem-selector>
  <button v-on:click="take_gems()"
          v-bind:disabled="!any_gems_selected && false">
    take gems
  </button>
</div>
`
});

Vue.component('gem-selector', {
    props: ['supply_gems', 'gems', 'player_gems', 'player_cards'],
    computed: {
        can_increment: function() {
            var any_value_2 = false;
            var num_values_1 = 0;
            for (var i = 0; i < colours.length; i++) {
                var colour = colours[i];
                if (this.gems[colour] >= 2) {
                    any_value_2 = true;
                }
                if (this.gems[colour] == 1) {
                    num_values_1 += 1;
                }
            }
            var incrementable = {};
            for (var i = 0; i < colours.length; i++) {
                var colour = colours[i];
                incrementable[colour] = !any_value_2 && (
                    (num_values_1 == 1 && this.gems[colour] == 1 && this.supply_gems[colour] > 3) || 
                        ((num_values_1 < 3) && this.supply_gems[colour] > 0 && this.gems[colour] == 0));
            }
            return incrementable;
        },
        can_decrement: function() {
            var decrementable = {};
            for (var i = 0; i < colours.length; i++) {
                var colour = colours[i];
                decrementable[colour] = (this.gems[colour] > 0);
            }
            return decrementable;
        },
        show_button: function() {
            let show = {};
            for (let colour of colours) {
                if (this.supply_gems[colour] > 0) {
                    show[colour] = true;
                } else {
                show[colour] = false;
                }
            }
            show['gold'] = false;
            return show;
        }
    },
    template: `
<table class="gem-selector">
  <tr style="margin-bottom:15px">
    <td>current gems</td>
    <gems-table-gem-counter v-for="(number, colour) in player_gems"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-gem-counter>
  </tr>
  <tr style="margin-bottom:15px">
    <td>current cards</td>
    <gems-table-card-counter v-for="(number, colour) in player_cards"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-card-counter>
  </tr>
  <tr><td style="height:7px"></td></tr>
  <tr>
    <td>gems gained</td>
    <gems-table-gem-counter v-for="(number, colour) in gems"
        v-bind:key="colour"
        v-bind:colour="colour"
        v-bind:number="number">
    </gems-table-gem-counter>
  </tr>
  <tr>
    <td></td>
    <increment-button v-for="(number, colour) in gems"
                      v-bind:key="colour"
                      v-bind:enabled="can_increment[colour]"
                      v-bind:show_button="show_button[colour]"
                      v-on:increment="gems[$event] += 1"
                      v-bind:colour="colour">
    </increment-button>
  </tr>
  <tr>
    <td></td>
    <decrement-button v-for="(number, colour) in gems"
                      v-bind:key="colour"
                      v-bind:enabled="can_decrement[colour]"
                      v-bind:show_button="show_button[colour]"
                      v-on:decrement="gems[$event] -= 1"
                      v-bind:colour="colour">
    </decrement-button>
  </tr>
</table>
`
});

Vue.component('increment-button', {
    props: ['colour', 'enabled', 'show_button'],
    computed: {
        show: function() {
            if (this.show_button) {
                return 1;
            }
            return 0;
        }
    },
    template: `
<td class="increment-button">
  <button v-bind:disabled="!enabled"
          v-bind:style="{opacity:show}"
          v-on:click="$emit('increment', colour)">
    +
  </button>
</td>
`
});

Vue.component('decrement-button', {
    props: ['colour', 'enabled', 'show_button'],
    computed: {
        show: function() {
            if (this.show_button) {
                return 1;
            }
            return 0;
        }
    },
    template: `
<td class="decrement-button">
  <button v-bind:disabled="!enabled"
          v-bind:style="{opacity:show}"
          v-on:click="$emit('decrement', colour)">
    -
  </button>
</td>
`
});

Vue.component('player-display', {
    props: ['player', 'is_current_player', 'can_show_card_buttons', 'is_human'],
    computed: {
        player_num_gems: function() {
            return (this.player.gems['white'] +
                    this.player.gems['blue'] +
                    this.player.gems['green'] +
                    this.player.gems['red'] +
                    this.player.gems['black'] +
                    this.player.gems['gold']);
        },
        border_width: function() {
            if (this.is_current_player) {
                return "6px";
            }
            return "5px";
        },
        border_colour: function() {
            if (this.is_current_player) {
                return 'green';
            }
            return '#bbeebb';
        },
        background_colour: function() {
            if (this.is_current_player) {
                return '#eeffee';
            }
            return '#ffffee';
        },
        player_type: function() {
            // Access the Vue root instance to get the bot types
            const app = this.$root;
            
            if (this.is_human) {
                return 'human';
            } else if (app.game_mode === 'bot_vs_bot') {
                // In bot vs bot mode, show which bot type
                const botType = this.player.number === 1 ? app.player1_bot_type : app.player2_bot_type;
                
                switch(botType) {
                    case 'neural':
                        return 'Neural Bot';
                    case 'greedy':
                        return 'Greedy Bot';
                    case 'random':
                        return 'Random Bot';
                    case 'aggro':
                        return 'Aggro Bot';
                    case 'ppo':
                        return 'PPO Bot';
                    default:
                        return 'AI';
                }
            } else {
                return 'AI';
            }
        },
        player_class: function() {
            return {
                'player-display': true,
                'current-player': this.is_current_player
            };
        },
        show_card_buttons: function() {
            if (!this.is_current_player) {
                return false;
            }
            if (!this.is_human) {
                return false;
            }
            return this.is_current_player;
        }
    },
    template: `
<div v-bind:class="player_class"
     v-bind:style="{borderWidth: border_width,borderColor: border_colour,backgroundColor: background_colour}">
<h3>P{{ player.number }} ({{ player_type }}): {{ player.score }} points, {{ player_num_gems }} gems</h3>
    <gems-table v-bind:gems="player.gems"
                v-bind:show_card_count="true"
                v-bind:cards="player.card_colours">
    </gems-table>
    <cards-display v-show="player.cards_in_hand.length > 0"
                   v-bind:cards="player.cards_in_hand"
                   v-bind:player="player"
                   v-bind:num_cards="3"
                   tier="hand"
                   v-bind:show_card_buttons="show_card_buttons"
                   v-bind:show_reserve_button="false"
                   style="height:180px;min-height:180px"
                   v-on:buy="$emit('buy', $event)">
    </cards-display>
    <nobles-display v-bind:nobles="player.nobles">
    </nobles-display>
</div>
`
});

Vue.component('cards-display', {
    props: ['cards', 'name', 'tier', 'player', 'show_reserve_button', 'num_cards', 'show_card_buttons'],
    methods: {
        reserve: function(card) {
            var card_index;
            for (var i = 0; i < this.cards.length; i++) {
                if (this.cards[i] === card) {
                    card_index = i;
                }
            }
            this.$emit('reserve', [this.tier, card_index]);
        },
        buy: function(card) {
            var card_index;
            for (var i = 0; i < this.cards.length; i++) {
                if (this.cards[i] === card) {
                    card_index = i;
                }
            }
            this.$emit('buy', [this.tier, card_index, this.player.can_afford(card)[1]]);
        }
    },
    computed: {
        card_width: function() {
            return ((100 - this.num_cards * 2) / this.num_cards).toString() + '%';
        }
    },
    template: `
<div class="cards-display">
    <h3>{{ name }}</h3>
    <ul class="single-line-list">
      <card-display
          v-bind:style="{width:card_width,maxWidth:card_width,minWidth:card_width}"
          v-for="card in cards"
          v-bind:show_reserve_button="show_reserve_button"
          v-bind:show_card_buttons="show_card_buttons"
          v-bind:player="player"
          v-bind:key="card.id"
          v-bind:card="card" 
          v-on:buy="buy($event)"
          v-on:reserve="reserve($event)">
      </card-display>
    </ul>
</div>
`
});

Vue.component('card-display', {
    props: ['card', 'player', 'show_reserve_button', 'show_card_buttons'],
    computed: {
        background_colour: function() {
            return background_colours[this.card.colour];
        },
        buyable: function() {
            return this.player.can_afford(this.card)[0];
        },
        reservable: function() {
            return (this.player.cards_in_hand.length < 3);
        },
        buy_button_top: function() {
            if (this.show_reserve_button) {
                return "26%";
            }
            return "5%";
        },
    },
    template: `
<li class="card-display">
<div class="card-display-contents" v-bind:style="{backgroundColor: background_colour}">
    <p class="card-points">{{ card.points }}</p>
    <button class="reserve-button"
            v-if="show_reserve_button && show_card_buttons && reservable"
            v-bind:disabled="!reservable"
            v-on:click="$emit('reserve', card)">
        reserve
    </button>
    <button class="buy-button"
            v-if="show_card_buttons && buyable"
            v-bind:disabled="!buyable"
            v-bind:style="{top: buy_button_top}"
            v-on:click="$emit('buy', card)">
        buy
    </button>
    <gems-list v-bind:gems="card.gems" 
               v-bind:display_zeros="false">
    </gems-list>
</div>
</li>
`
});


Vue.component('card-display-table-row', {
    props: ['colour', 'number', 'other'],
    template: `
<tr>
  <td>
    <gem-counter v-bind:colour="colour"
                 v-bind:number="number">
    </gem-counter>
  </td>
</tr>
`
});

Vue.component('supply-display', {
    props: ['gems', 'show_card_count', 'nobles'],
    computed: {
        num_gems: function() {
            let total = 0;
            for (let colour of colours) {
                total += this.gems[colour];
            }
            return total;
        }
    },
    template: `
<div class="supply-display">
    <h3>Supply: {{ num_gems }} coloured gems</h3>
    <gems-table v-bind:gems="gems"
                v-bind:show_card_count="show_card_count">
    </gems-table>
    <nobles-display v-bind:nobles="nobles">
    </nobles-display>
</div>
`
});

function noble_string(noble) {
    let string = '<span class="bold">3 points</span> for ';
    let first = true;
    for (let colour of colours) {
        if (colour in noble.cards && noble.cards[colour] > 0) {
            if (!first) {
                string += ', ';
            }
            string += '<span class="' + colour + '">' + noble.cards[colour] + ' ' + colour + '</span>';
            first = false;
        }
    }
    return string;
};

Vue.component('nobles-display', {
    props: ['nobles'],
    computed: {
        noble_strings: function() {
            let strings = [];
            for (let noble of this.nobles) {
                strings.push(noble_string(noble));
            }
            return strings
        },
    },
    template: `
<ul class="nobles-display">
    <li v-for="noble_string in noble_strings"><span v-html="noble_string"></span></li>
</ul>
`
});

Vue.component('ai-move-status', {
    props: ['player_index', 'num_possible_moves', 'ppo_bot_status'],
    mounted: function() {
        this.$emit('on_player_index')
    },
    template: `
<div class="ai-move-status">
    <div>
        <h3>AI Player {{ player_index + 1 }} thinking...</h3>
        <p>Evaluating {{ num_possible_moves }} possible moves</p>
        <p v-if="ppo_bot_status" class="ppo-error">{{ ppo_bot_status }}</p>
    </div>
</div>
`
});

Vue.component('winner-display', {
    props: ['winner_index', 'players'],
    computed: {
        winning_score: function () {
            if (this.winner_index === null) {
                return -1;
            }
            return this.players[this.winner_index].score;
        },
        winner_name: function() {
            if (this.winner_index === null) {
                return '';
            }
            
            // Get access to root Vue instance
            const app = this.$root;
            
            if (app.game_mode === 'bot_vs_bot') {
                // In bot vs bot mode, show which bot won
                const botType = this.winner_index === 0 ? app.player1_bot_type : app.player2_bot_type;
                let botName = '';
                
                switch(botType) {
                    case 'neural':
                        botName = 'Neural Bot';
                        break;
                    case 'greedy':
                        botName = 'Greedy Bot';
                        break;
                    case 'random':
                        botName = 'Random Bot';
                        break;
                    case 'aggro':
                        botName = 'Aggro Bot';
                        break;
                    case 'ppo':
                        botName = 'PPO Bot';
                        break;
                    default:
                        botName = 'AI';
                        break;
                }
                
                return `Player ${this.winner_index + 1} (${botName})`;
            } else {
                // In human vs AI mode, show if human or AI won
                if (app.human_player_indices.includes(this.winner_index)) {
                    return `Player ${this.winner_index + 1} (Human)`;
                } else {
                    return `Player ${this.winner_index + 1} (AI)`;
                }
            }
        }
    },
    template: `
<div class="winner-display">
    <h3>{{ winner_name }} wins with {{ winning_score }} points!
    </h3>

    <button v-on:click="$emit('reset')">
        play again
    </button>
</div>
`
});


function move_to_description(move) {
    let action = move.action;
    
    if (action === 'gems') {
        let gems_taken = [];
        for (let colour of all_colours) {
            if (move.gems[colour] > 0) {
                gems_taken.push(`${move.gems[colour]} ${colour}`);
            }
        }
        return `Took gems: ${gems_taken.join(', ')}`;
    } else if (action === 'reserve') {
        if (move.card) {
            return `Reserved a tier ${move.tier} ${move.card.colour} card worth ${move.card.points} points` + 
                   (move.gems.gold > 0 ? ` and received 1 gold gem` : ``);
        } else {
            return `Reserved a tier ${move.tier} card` + 
                   (move.gems.gold > 0 ? ` and received 1 gold gem` : ``);
        }
    } else if (action === 'buy_available') {
        if (move.card) {
            let cost_parts = [];
            for (let colour of all_colours) {
                if (move.gems[colour] > 0) {
                    cost_parts.push(`${move.gems[colour]} ${colour}`);
                }
            }
            const cost_string = cost_parts.length > 0 ? ` for ${cost_parts.join(', ')}` : ` for free`;
            return `Bought a tier ${move.tier} ${move.card.colour} card worth ${move.card.points} points${cost_string}`;
        } else {
            return `Bought a tier ${move.tier} card`;
        }
    } else if (action === 'buy_reserved') {
        if (move.card) {
            let cost_parts = [];
            for (let colour of all_colours) {
                if (move.gems[colour] > 0) {
                    cost_parts.push(`${move.gems[colour]} ${colour}`);
                }
            }
            const cost_string = cost_parts.length > 0 ? ` for ${cost_parts.join(', ')}` : ` for free`;
            return `Bought a reserved ${move.card.colour} card worth ${move.card.points} points${cost_string}`;
        } else {
            return `Bought a reserved card`;
        }
    }
    
    return "Unknown move";
}

Vue.component('moves-log-display', {
    props: ['moves'],
    computed: {
        move_strings: function () {
            let strs = [];
            for (let i = 0; i < this.moves.length; i++) {
                let move = this.moves[i];
                let round = math.floor(i / 2) + 1;
                let player = (i % 2) + 1;

                let description = move_to_description(move);
                strs.push({
                    html: '<span class="bold">R' + round + ' P' + player + ':</span> ' + description,
                    isLatest: i === this.moves.length - 1
                });
            }
            return strs;
        }
    },
    template: `
<div class="moves-log-display"
     v-if="moves.length > 0">
    <h3>Moves log:</h3>
    <ul>
        <li v-for="move in move_strings" 
            v-bind:class="{ 'latest-move': move.isLatest }">
            <span v-html="move.html"></span>
        </li>
    </ul>
</div>
`
});


function random_player_index() {
    if (math.random() > 0.5) {
        return 1;
    }
    return 0;
}


var app = new Vue({
    el: '#app',
    data: {
        state: test_state,
        human_player_indices: [random_player_index()],
        scheduled_move_func: null,
        discarding: false,
        winner_index: null,
        num_possible_moves: 0,
        debug_checked: false,
        game_mode: 'human_vs_ai',  // New: game mode selector
        bot_advance_mode: 'manual', // New: bot advancement mode
        auto_advance_delay: 0.1,   // New: delay for auto advancement
        last_move: null,           // New: store the last move for display
        player1_bot_type: 'neural', // New: bot type for player 1
        player2_bot_type: 'greedy', // New: bot type for player 2
        player_bot_instances: [],   // New: stores the bot instances for each player
        gems_selected: {'white': 0,
                        'blue': 0,
                        'green': 0,
                        'red': 0,
                        'black': 0,
                        'gold': 0},
        gems_discarded: {'white': 0,
                         'blue': 0,
                         'green': 0,
                         'red': 0,
                         'black': 0,
                         'gold': 0},
        ppo_bot_status: '',
        
        // Model configuration
        model_path: '',
        model_input_dim: 2300,
        model_output_dim: 100,
        model_status: '',
        model_status_class: '',
    },
    methods: {
        testChangeGems: function() {
            this.state.reduce_gems();
        },
        test_state_vector: function() {
            // for (let i = 0; i < 100; i++) {
            //     this.state.get_state_vector(0);
            // }
            console.log(this.state.get_state_vector());
            console.log('state vector length is', this.state.get_state_vector().length);

        },
        test_ai_move: function() {
            console.log('ai move:', ai.make_move(this.state));
        },
        test_change_player_type: function() {
            if (this.player_type === 'human') {
                this.player_type = 'ai';
            } else {
                this.player_type = 'human';
            }

            for (var i = 0; i < colours.length; i++) {
                this.gems_selected[colours[i]] = 0;
            }
        },
        test_moves: function() {
            console.log(this.state.get_valid_moves());
        },
        random_move: function() {
            let ai = new RandomAI();
            let move = ai.make_move(this.state);
            this.state.make_move(move);
        },
        test_win: function() {
            this.state.players[0].score = 15;
        },
        reset: function() {
            this.human_player_indices = [random_player_index()];
            this.state = new GameState();
            this.discarding = false;
            this.winner_index = null;
            this.last_move = null;

            for (let colour of all_colours) {
                this.gems_selected[colour] = 0;
            }
            
            if (this.game_mode === 'bot_vs_bot' || 
                (this.game_mode === 'human_vs_ai' && this.player_type === 'ai')) {
                if (this.game_mode === 'bot_vs_bot' && this.bot_advance_mode === 'auto') {
                    this.schedule_auto_ai_move();
                } else if (this.game_mode === 'human_vs_ai') {
                    this.schedule_ai_move();
                }
            }
        },
        on_player_index: function() {
            let winner = this.state.has_winner();
            if (!(winner === null)) {
                this.winner_index = winner;
                return;
            }

            if (!(this.scheduled_move_func === null)) {
                window.clearTimeout(this.scheduled_move_func);
                this.scheduled_move_func = null;
            }
            
            if (this.game_mode === 'bot_vs_bot') {
                if (this.bot_advance_mode === 'auto') {
                    this.schedule_auto_ai_move();
                }
                // In manual mode, we wait for user to click "Next Move"
            } else if (this.game_mode === 'human_vs_ai' && this.player_type === 'ai') {
                this.schedule_ai_move();
            }

            // Reset the gems selected gui
            for (let colour of all_colours) {
                this.gems_selected[colour] = 0;
            }
        },
        schedule_ai_move: function() {
            this.num_possible_moves = this.state.get_valid_moves(this.state.current_player_index).length;
            window.setTimeout(this.do_ai_move, 50);
        },
        validateGameState: function() {
            console.group("Game State Validation");
            let isValid = true;
            let errors = [];
            
            try {
                // Check that the current player index is valid
                if (this.state.current_player_index < 0 || this.state.current_player_index >= this.state.players.length) {
                    errors.push(`Invalid current player index: ${this.state.current_player_index}`);
                    isValid = false;
                }
                
                // Check that players exist
                if (!this.state.players || this.state.players.length === 0) {
                    errors.push("No players in game state");
                    isValid = false;
                }
                
                // Check if market cards are valid
                if (!this.state.tier_1_visible || !Array.isArray(this.state.tier_1_visible)) {
                    errors.push("Tier 1 visible cards invalid");
                    isValid = false;
                }
                if (!this.state.tier_2_visible || !Array.isArray(this.state.tier_2_visible)) {
                    errors.push("Tier 2 visible cards invalid");
                    isValid = false;
                }
                if (!this.state.tier_3_visible || !Array.isArray(this.state.tier_3_visible)) {
                    errors.push("Tier 3 visible cards invalid");
                    isValid = false;
                }
                
                // Check if supply gems are valid
                if (!this.state.supply_gems) {
                    errors.push("Supply gems are invalid");
                    isValid = false;
                } else {
                    for (let color of all_colours) {
                        if (this.state.supply_gems[color] === undefined) {
                            errors.push(`Missing ${color} in supply gems`);
                            isValid = false;
                        }
                    }
                }
                
                // Check each player's state
                this.state.players.forEach((player, idx) => {
                    // Check if player has gems
                    if (!player.gems) {
                        errors.push(`Player ${idx + 1} has invalid gems`);
                        isValid = false;
                    }
                    
                    // Check if player has cards played
                    if (!player.cards_played || !Array.isArray(player.cards_played)) {
                        errors.push(`Player ${idx + 1} has invalid cards_played`);
                        isValid = false;
                    }
                    
                    // Check if player has cards in hand
                    if (!player.cards_in_hand || !Array.isArray(player.cards_in_hand)) {
                        errors.push(`Player ${idx + 1} has invalid cards_in_hand`);
                        isValid = false;
                    }
                    
                    // Check if player has valid score
                    if (typeof player.score !== 'number') {
                        errors.push(`Player ${idx + 1} has invalid score: ${player.score}`);
                        isValid = false;
                    }
                });
                
                // Try to get valid moves as a final sanity check
                try {
                    const moves = this.state.get_valid_moves();
                    console.log(`Valid moves: ${moves.length}`);
                    if (moves.length === 0) {
                        errors.push("No valid moves available");
                        // This may not be an error (could be game over)
                        console.warn("WARNING: No valid moves, but this could be normal at game end");
                    }
                } catch (e) {
                    errors.push(`Error getting valid moves: ${e.message}`);
                    isValid = false;
                }
                
                // Log validation results
                if (isValid) {
                    console.log("✅ Game state appears valid");
                } else {
                    console.error("❌ Game state validation failed:", errors);
                    console.error("Game state dump:", JSON.parse(JSON.stringify(this.state)));
                }
                
                return isValid;
            } catch (e) {
                console.error("Error during game state validation:", e);
                console.error("Game state dump:", this.state);
                console.groupEnd();
                return false;
            }
            
            console.groupEnd();
        },
        nn_ai_move: function() {
            console.group("AI Move");
            console.log('Starting AI move');
            
            // Validate game state before attempting to make a move
            if (!this.validateGameState()) {
                console.error("Invalid game state detected, attempting reset to safe state");
                // Try to proceed by incrementing player or other recovery
                try {
                    this.state.increment_player();
                    console.log("Incremented player as recovery");
                } catch (e) {
                    console.error("Recovery failed:", e);
                    // If all else fails, reset the game
                    if (confirm("Game state is corrupted. Reset the game?")) {
                        this.reset();
                    }
                }
                console.groupEnd();
                return;
            }
            
            // Use the appropriate bot for the current player
            const botType = this.state.current_player_index === 0 ? this.player1_bot_type : this.player2_bot_type;
            console.log(`Using ${botType} bot for player ${this.state.current_player_index + 1}`);
            
            if (botType === 'ppo') {
                // Use the asynchronous approach for PPO bot
                this.fetchPPOBotMove();
            } else {
                // For other bots, use the existing synchronous approach
                const bot = this.getBotForCurrentPlayer();
                let move = bot.make_move(this.state);
                console.log('AI move is', move);
                if (!move) {
                    console.error("Bot returned null/undefined move");
                    console.groupEnd();
                    return;
                }
                
                this.last_move = move;
                this.last_move_description = move_to_description(move);
                
                try {
                    this.state.make_move(move);
                    console.log("Move applied successfully");
                } catch (e) {
                    console.error("Error applying move:", e);
                    console.groupEnd();
                    return;
                }
            }
            console.groupEnd();
        },
        fetchPPOBotMove: function() {
            console.group("PPO Bot Move - Start");
            console.log("Current player:", this.state.current_player_index + 1);
            
            const moves = this.state.get_valid_moves();
            console.log(`Valid moves: ${moves.length}`);
            
            // Show a loading indicator or message
            this.num_possible_moves = moves.length;
            this.ppo_bot_status = 'Thinking...';
            
            // Safety check - if no valid moves, we can't proceed
            if (moves.length === 0) {
                console.error("CRITICAL ERROR: No valid moves available for PPO bot!");
                this.ppo_bot_status = 'Error: No valid moves available!';
                console.groupEnd();
                return;
            }
            
            // Create a timeout to prevent getting stuck
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => {
                    console.error("FETCH TIMEOUT: API call took too long (15s)");
                    reject(new Error('Fetch timeout after 15 seconds'));
                }, 15000);
            });
            
            // Prepare state data for API
            try {
                const state_json = JSON.stringify({
                    current_player_index: this.state.current_player_index,
                    players: this.state.players,
                    supply_gems: this.state.supply_gems,
                    tier_1_visible: this.state.tier_1_visible,
                    tier_2_visible: this.state.tier_2_visible,
                    tier_3_visible: this.state.tier_3_visible,
                    nobles: this.state.nobles,
                    moves: moves
                });
                
                // Make the API call with timeout
                console.log('Fetching PPO bot move from server...');
                
                Promise.race([
                    fetch('http://localhost:5000/api/ppo_move', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: state_json
                    }),
                    timeoutPromise
                ])
                .then(response => {
                    console.log(`Server response status: ${response.status}`);
                    if (!response.ok) {
                        return response.json().then(errorData => {
                            throw new Error(errorData.error || `API request failed with status: ${response.status}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('API response:', data);
                    
                    // Check if the response contains an error
                    if (data.error) {
                        throw new Error(`Server error: ${data.error}`);
                    }
                    
                    const move_index = data.move_index;
                    console.log('PPO server responded with move index:', move_index);
                    
                    // Make sure the move index is valid
                    if (move_index >= 0 && move_index < moves.length) {
                        const move = moves[move_index];
                        console.log('PPO bot move is', move);
                        this.last_move = move;
                        this.last_move_description = move_to_description(move);
                        try {
                            console.log("Applying move:", JSON.parse(JSON.stringify(move)));
                            this.state.make_move(move);
                            console.log("Move applied successfully");
                            this.ppo_bot_status = '';
                            
                            // Check for winner after making the move
                            this.checkWinnerAfterMove();
                            
                            // Schedule next move if appropriate
                            this.handleNextMoveScheduling();
                        } catch (moveError) {
                            console.error("CRITICAL ERROR: Failed to apply move:", moveError);
                            console.error("Move that failed:", JSON.parse(JSON.stringify(move)));
                            this.ppo_bot_status = `Error applying move: ${moveError.message}`;
                        }
                    } else {
                        console.error('Invalid move index returned from API:', move_index, 'Max valid index:', moves.length - 1);
                        this.ppo_bot_status = `Server returned invalid move index: ${move_index}`;
                    }
                    console.groupEnd();
                })
                .catch(error => {
                    console.error('Error getting PPO move from server:', error);
                    console.error('Error type:', error.constructor.name);
                    console.error('Error stack:', error.stack);
                    this.ppo_bot_status = `PPO Bot Error: ${error.message}`;
                    console.groupEnd();
                });
            } catch (error) {
                console.error("CRITICAL ERROR during fetch setup:", error);
                console.error("Error stack:", error.stack);
                this.ppo_bot_status = `Setup error: ${error.message}`;
                console.groupEnd();
            }
        },
        checkWinnerAfterMove: function() {
            const winner = this.state.has_winner();
            if (winner !== null) {
                this.winner_index = winner;
                console.log(`Player ${winner + 1} has won the game!`);
                
                // Cancel any pending auto moves
                if (this.scheduled_move_func !== null) {
                    window.clearTimeout(this.scheduled_move_func);
                    this.scheduled_move_func = null;
                }
            }
        },
        handleNextMoveScheduling: function() {
            // Auto-schedule next move if we're in bot vs bot auto mode and there's no winner
            if (this.game_mode === 'bot_vs_bot' && 
                this.bot_advance_mode === 'auto' && 
                this.winner_index === null) {
                this.schedule_auto_ai_move();
            }
            
            // Update the UI to highlight the current player
            if (this.winner_index === null) {
                this.highlight_current_player();
            }
        },
        do_ai_move: function() {
            this.scheduled_move_func = null;
            if (this.player_type === 'ai') {
                const botType = this.state.current_player_index === 0 ? this.player1_bot_type : this.player2_bot_type;
                
                if (botType === 'ppo') {
                    // For PPO bot, use the specific fetchPPOBotMove method
                    this.fetchPPOBotMove();
                } else {
                    // For other bots, use the standard approach
                    this.nn_ai_move();
                    
                    // For non-PPO bots, we need to check for winner here
                    // (PPO bot handles this in its own callback)
                    this.checkWinnerAfterMove();
                    
                    // Schedule next move if appropriate
                    if (this.game_mode === 'bot_vs_bot' && 
                        this.bot_advance_mode === 'auto' && 
                        this.winner_index === null) {
                        this.schedule_auto_ai_move();
                    }
                }
            }
        },
        do_move_gems: function(info) {
            let passed_gems = {};
            for (let colour of all_colours) {
                if (colour in this.gems_selected) {
                    passed_gems[colour] = this.gems_selected[colour];
                }
            }
            this.state.make_move({action: 'gems',
                                  gems: passed_gems},
                                 false);
            for (let colour of colours) {
                this.gems_selected[colour] = 0;
            }

            this.check_if_discarding();
        },
        do_move_reserve: function(info) {
            var gold_taken = 0;
            if (this.state.supply_gems['gold'] > 0) {
                gold_taken = 1;
            }
            this.state.make_move({action: 'reserve',
                                  tier: info[0],
                                  index: info[1],
                                  gems: {'gold': gold_taken}},
                                 false);
            this.check_if_discarding();
        },
        do_move_buy: function(info) {
            let tier = info[0];
            if (tier === 'hand') {
                this.state.make_move({action: 'buy_reserved',
                                      tier: info[0],
                                      index: info[1],
                                      gems: info[2]},
                                     false);
            } else {
                this.state.make_move({action: 'buy_available',
                                      tier: info[0],
                                      index: info[1],
                                      gems: info[2]},
                                     false);
            }
            this.check_if_discarding();
        },
        check_if_discarding: function() {
            // let player = this.human_player;
            let player = this.current_player;
            if (player.total_num_gems() > 10) {
                this.discarding = true;
            } else {
                this.discarding = false;
                this.state.increment_player();
                
                // Check if there's a winner after incrementing player
                const winner = this.state.has_winner();
                if (winner !== null) {
                    this.winner_index = winner;
                    console.log(`Player ${winner + 1} has won the game!`);
                    
                    // Cancel any pending auto moves
                    if (this.scheduled_move_func !== null) {
                        window.clearTimeout(this.scheduled_move_func);
                        this.scheduled_move_func = null;
                    }
                }
            }
        },
        do_discard_gems: function() {
            let player = this.current_player;
            let state = this.state;
            for (let colour of all_colours) {
                // player.gems[colour] -= this.gems_discarded[colour];
                state.supply_gems[colour] += this.gems_discarded[colour];
                this.gems_discarded[colour] = 0;
            }
            this.check_if_discarding();
        },
        // Manual advancement for bot vs bot mode
        do_manual_ai_move: function() {
            console.group("Manual AI Move");
            console.log("Manual next move button clicked");
            
            // Validate game state first
            if (!this.validateGameState()) {
                console.error("Invalid game state detected when clicking Next Move");
                if (confirm("Game state is corrupted. Reset the game?")) {
                    this.reset();
                }
                console.groupEnd();
                return;
            }
            
            try {
                this.do_ai_move();
                
                // Only highlight if there's no winner yet and it's not a PPO bot
                // (PPO bot handles highlighting in its own callback)
                const botType = this.state.current_player_index === 0 ? this.player1_bot_type : this.player2_bot_type;
                if (this.winner_index === null && botType !== 'ppo') {
                    this.highlight_current_player();
                }
            } catch (e) {
                console.error("Error during manual AI move:", e);
                console.error("Error stack:", e.stack);
                alert("An error occurred during the AI move. Check the console for details.");
            }
            
            console.groupEnd();
        },
        
        // Auto advancement for bot vs bot mode
        schedule_auto_ai_move: function() {
            // Don't schedule new moves if there's a winner
            if (this.winner_index !== null) {
                return;
            }
            
            // Clear any existing scheduled move to prevent multiple timers
            if (this.scheduled_move_func !== null) {
                window.clearTimeout(this.scheduled_move_func);
                this.scheduled_move_func = null;
            }
            
            // Get the current number of possible moves for display
            this.num_possible_moves = this.state.get_valid_moves(this.state.current_player_index).length;
            
            // Check if current player is using the PPO bot
            const botType = this.state.current_player_index === 0 ? this.player1_bot_type : this.player2_bot_type;
            
            // Schedule the next move with a timeout
            this.scheduled_move_func = window.setTimeout(() => {
                // Execute the AI move - special handling for PPO bot
                if (botType === 'ppo') {
                    this.fetchPPOBotMove();
                } else {
                    this.do_ai_move();
                }
                
                // Only highlight if there's no winner
                if (this.winner_index === null) {
                    this.highlight_current_player();
                }
            }, this.auto_advance_delay * 1000);
        },
        
        // Highlight the current player's section
        highlight_current_player: function() {
            // Remove existing highlights
            document.querySelectorAll('.move-highlight').forEach(el => {
                el.classList.remove('move-highlight');
            });
            
            // Add highlight to current player
            const playerElements = document.querySelectorAll('.player-display');
            if (playerElements && playerElements.length > this.state.current_player_index) {
                playerElements[this.state.current_player_index].classList.add('move-highlight');
            }
            
            // Also highlight the last move in the log if available
            if (this.last_move) {
                setTimeout(() => {
                    const moveLogItems = document.querySelectorAll('.moves-log-display li');
                    if (moveLogItems && moveLogItems.length > 0) {
                        const lastMoveItem = moveLogItems[moveLogItems.length - 1];
                        lastMoveItem.classList.add('move-highlight');
                    }
                }, 100); // Small delay to ensure the DOM has updated
            }
        },
        
        // Get the appropriate bot instance for the current player
        getBotForCurrentPlayer: function() {
            const playerIndex = this.state.current_player_index;
            const botType = playerIndex === 0 ? this.player1_bot_type : this.player2_bot_type;
            
            console.log('current player bot type is', botType);
            switch (botType) {
            case 'neural':
                // Make sure we always pass the state_vector_v02 function
                return ai_neural || new NeuralNetAI('', state_vector_v02);
            case 'greedy':
                return new GreedyAI();
            case 'random':
                return new RandomAI();
            case 'aggro':
                return new AggroAI();
            case 'ppo':
                return 'ppo'; // Keep 'ppo' for backward compatibility, but this now uses the ModelBotAI
            }
            console.log('unrecognised bot type', botType);
            return 'human';
        },
        
        // Load a new model from the specified path
        loadModel: function() {
            this.model_status = 'Loading...';
            this.model_status_class = 'loading';
            
            const modelConfig = {
                model_path: this.model_path,
                input_dim: this.model_input_dim,
                output_dim: this.model_output_dim
            };
            
            fetch('/api/set_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(modelConfig)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.model_status = 'Success: ' + data.message;
                    this.model_status_class = 'success';
                } else {
                    this.model_status = 'Error: ' + (data.error || 'Unknown error');
                    this.model_status_class = 'error';
                }
            })
            .catch(error => {
                console.error('Error loading model:', error);
                this.model_status = 'Error: ' + error.message;
                this.model_status_class = 'error';
            });
        },
    },
    computed: {
        player_type: function() {
            if (this.game_mode === 'human_vs_ai') {
                for (let index of this.human_player_indices) {
                    if (index === this.state.current_player_index) {
                        return 'human';
                    }
                }
                return 'ai';
            } else {
                // In bot_vs_bot mode, all players are AI
                return 'ai';
            }
        },
        current_player: function() {
            return this.state.players[this.state.current_player_index];
        },
        human_player: function() {
            return this.state.players[this.human_player_index];
        },
        round_number: function() {
            return this.state.round_number;
        },
        supply_gems: function() {
            return this.state.supply_gems;
        },
        players: function() {
            return this.state.players;
        },
        indexed_players: function() {
            var players = {};
            for (let i = 0; i < this.num_players; i++) {
                players[i] = this.players[i];
            }
            return players;
        },
        show_card_buttons: function() {
            return this.game_mode === 'human_vs_ai' && this.player_type === 'human';
        },
        has_winner: function() {
            return !(this.winner_index === null);
        },
        last_move_description: {
            get: function() {
                return this.last_move ? move_to_description(this.last_move) : '';
            },
            set: function(newValue) {
                // This is needed for two-way binding
                this._last_move_description = newValue;
            }
        }
    },
    watch: {
        // Watch for changes in game mode
        game_mode: function(newMode) {
            this.reset();
        },
        
        // Watch for changes in bot advance mode
        bot_advance_mode: function(newMode) {
            if (this.game_mode === 'bot_vs_bot') {
                if (newMode === 'auto') {
                    this.schedule_auto_ai_move();
                } else {
                    // Cancel any scheduled auto moves
                    if (this.scheduled_move_func !== null) {
                        window.clearTimeout(this.scheduled_move_func);
                        this.scheduled_move_func = null;
                    }
                }
            }
        },
        
        // Watch for changes in bot types
        player1_bot_type: function(newType) {
            if (this.game_mode === 'bot_vs_bot') {
                // Reset the game when bot type changes
                this.reset();
            }
        },
        
        player2_bot_type: function(newType) {
            if (this.game_mode === 'bot_vs_bot') {
                // Reset the game when bot type changes
                this.reset();
            }
        }
    },
    mounted: function() {
        // Initialize with the selected game mode
        this.reset();
        
        // If we're in bot vs bot auto mode, start it right away
        if (this.game_mode === 'bot_vs_bot' && this.bot_advance_mode === 'auto') {
            this.schedule_auto_ai_move();
        }
    }
});


if (app.human_player_indices[0] != 0) {
    app.schedule_ai_move();
}
