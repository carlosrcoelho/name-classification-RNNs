import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, random_training_example, line_to_tensor, letter_to_tensor

# define the RNN model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size  # it means the number of hidden units
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # input to output
        self.softmax = nn.LogSoftmax(dim=1)  # it means the dimension along which Softmax will be computed

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)  # Concatenates the given sequence of seq tensors in the given dimension.

        hidden = self.i2h(combined)  # input to hidden
        output = self.i2o(combined)  # input to output
        output = self.softmax(output)  # softmax
        return output, hidden
    
    def init_hidden(self):  # initialize the hidden layer
        return torch.zeros(1, self.hidden_size)   # it means the size of hidden layer is 1 * hidden_size

# load data 
category_lines, all_categories = load_data()
n_categories = len(all_categories)
# print('# categories:', n_categories, all_categories)

# define the model
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)  # it means the input size is N_LETTERS, the hidden size is 128, the output size is n_categories



# one step as an example (without training)
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()
output, next_hidden = rnn(input_tensor, hidden_tensor)
# print(output.size(), next_hidden.size())

# whole sequence as an example (without training)
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()
output, next_hidden = rnn(input_tensor[0], hidden_tensor)
# print(output.size(), next_hidden.size())

def category_from_output(output):  # it means the output is the probability of each category 
    category_idx = torch.argmax(output).item()  # Returns the indices of the maximum value of all elements in the input tensor.
    return all_categories[category_idx]

print(category_from_output(output))  # probability of each category (without training)


# START TRAINING

# define the loss function
criterion = nn.NLLLoss()  # The negative log likelihood loss. It is useful to train a classification problem with C classes.
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)  # Stochastic gradient descent

def train(category_tensor, line_tensor):  # it means the category_tensor is the probability of each category, the line_tensor is the input tensor
    hidden = rnn.init_hidden()  # initialize the hidden layer

    for i in range(line_tensor.size()[0]):  # line_tensor.size()[0] means the length of line_tensor
        output, hidden = rnn(line_tensor[i], hidden)  # it means the output is the probability of each category, the hidden is the hidden layer

    loss = criterion(output, category_tensor)  # calculate the loss
    optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.
    loss.backward()  # it means the back propagation
    optimizer.step()  # Performs a single optimization step (parameter update).

    return output, loss.item()  # Returns the value of this tensor as a standard Python number. This only works for tensors with one element. For other cases, see tolist().

# traning loop

n_iters = 100000
plot_steps, print_steps = 1000, 5000
current_loss = 0
all_losses = []

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)  # it means the category is the category, the line is the input, the category_tensor is the probability of each category, the line_tensor is the input tensor
    output, loss = train(category_tensor, line_tensor)  # it means the output is the probability of each category, the loss is the loss
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = 'CORRECT' if guess == category else f'WRONG ({category})'
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

# plot the loss
plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():  # Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
        print(guess)

while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    predict(sentence)