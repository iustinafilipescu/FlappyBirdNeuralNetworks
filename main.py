import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird


size_input = 4
# our state: tuple of the distance on the x and y axis between the bird and the upcoming pipes,
# the velocity and the distance between the next pipe and the next next one on the y axis
size_hidden_layer = 100
# weights = [np.random.normal(scale=1 / np.sqrt(size_input), size=(size_hidden_layer, size_input)),
#                np.random.normal(scale=1 / np.sqrt(size_hidden_layer), size=(2, size_hidden_layer))]
# biases = [np.random.randn(size_hidden_layer, 1), np.random.randn(2, 1)]
weights = np.load("weights.txt.npy", allow_pickle=True)
biases = np.load("biases.txt.npy", allow_pickle=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(np.negative(z).astype(float)))


def sigmoid_derivated(z):
    return sigmoid(z) * (1 - sigmoid(z))


def train_feedforward(batch_input):
    last_layer_output = batch_input
    y = [last_layer_output]
    z = []

    for i in range(2):
        layer_sum = np.dot(weights[i], last_layer_output)
        sum1 = []
        for j in range(len(layer_sum)):
            sum1.append(layer_sum[j] + biases[i][j])
        z.append(sum1)
        last_layer_output = layer_sum
        y.append(sum1)
    return z, y


def train_backpropagation(t, z, y):
    add_weights = [np.zeros(weight.shape) for weight in weights]
    add_biases = [np.zeros(bias.shape) for bias in biases]

    # Cross entropy
    y = np.array(y, dtype=object)
    t = np.array(t, dtype=object)
    delta = y[-1] - t
    # transposing the y matrix
    y0 = list(y[0])
    y0.reverse()
    y1 = y[1]
    y1 = [list(i) for i in zip(*y1)]
    y2 = y[2]
    y2 = [list(i) for i in zip(*y2)]
    y = [[y0], y1, y2]
    # Backpropagation
    add_biases[-1] = delta
    add_weights[-1] = np.dot(delta, y[-2])
    for i in range(2, len(weights) + 1):
        delta = np.dot(weights[-i + 1].transpose(), delta) * sigmoid_derivated(z[-i])
        add_biases[-i] = delta
        add_weights[-i] = np.dot(delta, y[-i - 1])

    return add_weights, add_biases


def training(nr_iterations, batch_size, batch_count, l2_constant):
    p.reset_game()
    score_max = 0
    score = 0
    for _ in range(nr_iterations):
        print("Iteratie")
        print(_)
        for i in range(batch_count):
            print("batch")
            print(i)
            weights_update = [np.zeros(weight.shape) for weight in weights]
            biases_update = [np.zeros(bias.shape) for bias in biases]
            for j in range(batch_size):
                if p.game_over():
                    score = 0
                    p.reset_game()
                state = p.getGameState()
                batch_input = (state["next_pipe_dist_to_player"], state["player_y"]-state["next_pipe_bottom_y"],
                               state["player_vel"], state["next_pipe_bottom_y"]-state["next_next_pipe_bottom_y"])  #normalizare pentru fiecare feature in parte
                #first feedforward
                z, y = train_feedforward(batch_input)
                #choosing the action with the higher Q value
                reward = p.act(np.argmax(z[-1])*119)
                if reward == 21.0:
                    score = score+1
                if score > score_max:
                    score_max = score
                #doing the second feedforward to get the max Q(s',a')
                statePrim = p.getGameState()
                batch_input = (statePrim["next_pipe_dist_to_player"], statePrim["player_y"]-statePrim["next_pipe_bottom_y"],
                               statePrim["player_vel"], statePrim["next_pipe_bottom_y"]-statePrim["next_next_pipe_bottom_y"])
                z1, y1 = train_feedforward(batch_input)
                Qmax = np.max(z1[-1])
                iMax = np.argmax(z[-1])
                output = z[-1]
                output[iMax] += alpha * (reward + gamma * Qmax - output[iMax])
                #Backpropagation
                add_weights, add_biases = train_backpropagation(output, z, y)
                for k in range(len(add_weights)):
                    weights_update[k] = weights_update[k] + add_weights[k]
                    biases_update[k] = biases_update[k] + add_biases[k]
            for k in range(len(weights_update)):
                #using the L2 regulation
                weights[k] = weights[k] - (alpha / batch_size) * weights_update[k] - alpha * (l2_constant / batch_size*batch_count) * weights[k]
                biases[k] = biases[k] - (alpha / batch_size) * biases_update[k]
    print("scor:")
    print(score_max)


def testing(dead):
    score_max = 0
    score = 0
    p.reset_game()
    i = 0
    while i < dead:
        if p.game_over():
            score = 0
            p.reset_game()
            i += 1
        state = p.getGameState()
        batch_input = (
        state["next_pipe_dist_to_player"], state["player_y"] - state["next_pipe_bottom_y"], state["player_vel"],
        state["next_pipe_bottom_y"] - state["next_next_pipe_bottom_y"])

        z, y = train_feedforward(batch_input)
        reward = p.act(np.argmax(z[-1]) * 119)
        if reward == 21.0:
            score = score + 1
        if score > score_max:
            score_max = score
    print(score_max)


class NaiveAgent:
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]

###################################
game = FlappyBird(
    height=512, width=288
)  # create our game

fps = 30  # fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = 100

# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen, reward_values={"positive": 20.0, "tick": 0.5, "loss": -1000.0})

# our Naive agent!
agent = NaiveAgent(p.getActionSet())


# init agent and game.
p.init()

# lets do a random number of NOOP's
for i in range(np.random.randint(0, max_noops)):
    reward = p.act(p.NOOP)


alpha = 0.01
gamma = 0.95
q_table = {}
episodes = []
reward = 0.0

# start our training loop
# for f in range(nb_frames):
#     # if the game is over
#     if p.game_over():
#         p.reset_game()
#
#     obs = p.getScreenRGB()
#     action = agent.pickAction(reward, obs)
#
#     state=game.getGameState()
#     state1=(state["next_pipe_dist_to_player"],state["player_y"]-state["next_pipe_bottom_y"],state["player_vel"],
#     state["next_pipe_bottom_y"]-state["next_next_pipe_bottom_y"])
#     reward = p.act(action)
#     if action==119:
#         action=1
#     else:
#         action=0
#
#     stateStr=str(state["next_pipe_dist_to_player"])+"_"+str(state["player_y"]-state["next_pipe_bottom_y"])+"_"+str(state["player_vel"])+"_"+str(state["next_pipe_bottom_y"]-state["next_next_pipe_bottom_y"])
#     state=game.getGameState()
#
#     state2=(state["next_pipe_dist_to_player"],state["player_y"]-state["next_pipe_bottom_y"],state["player_vel"],state["next_pipe_bottom_y"]-state["next_next_pipe_bottom_y"])
#
#     state1Str = str(state1[0]) + str(state1[1]) + str(state1[2]) + str(state1[3])
#     state2Str = str(state2[0]) + str(state2[1]) + str(state2[2]) + str(state2[3])
#     if state1Str not in q_table:
#         q_table[state1Str] = [0, 0]
#     else:
#         if state2Str in q_table:
#             q_table[state1Str][action] = q_table[state1Str][action] + alpha * (reward) + gamma * max(
#                 q_table[state2Str][0:2])
#         else:
#             q_table[state1Str][action] = q_table[state1Str][action] + alpha * (reward)
#
#
#     episodes.append((state1, action, reward, state2))


# print(training(4,50,10,5))
print(testing(100))

np.save("weights.txt", weights)
np.save("biases.txt", biases)
