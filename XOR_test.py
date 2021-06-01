import numpy as np
from MLP import MLP

log = open("xortest.txt", "w")
print("Q1 XOR TEST\n", file=log)


def XOR(max_epochs, learning_rate, NH):
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    NI = 2
    NO = 1
    NN = MLP(NI, NH, NO)

    NN.randomise()

    print('Epochs = ' + str(max_epochs))
    print('Epochs = ' + str(max_epochs), file=log)

    print('Learning rate = ' + str(learning_rate))
    print('Learning rate = ' + str(learning_rate), file=log)

    print('Hidden units = ' + str(NH))
    print('Hidden units = ' + str(NH) + '\n\n', file=log)

    # training
    for i in range(max_epochs):
        NN.forward(inputs, 'sigmoid')
        error = NN.backward(inputs, outputs, 'sigmoid')
        NN.updateWeights(learning_rate)

        if (i + 1) == 100 or (i + 1) == 1000 or (i + 1) == 10000 or (i + 1) == 100000 or (i + 1) == 1000000:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=log)

    # get accuracy after training
    accuracy = float()
    for i in range(len(inputs)):
        NN.forward(inputs[i], 'sigmoid')

        if (outputs[i][0] == 0):
            accuracy += 1 - NN.O[0]
        elif (outputs[i][0] == 1):
            accuracy += NN.O[0]
    print('\nAccuracy = ' + str(accuracy / 4), file=log)


epochs = [100000]
learning_rate = [1.0, 0.75, 0.5,0.25, .05]
num_hidden = 6

for i in range(len(epochs)):
    for j in range(len(learning_rate)):
        XOR(epochs[i], learning_rate[j], num_hidden)
        print('\n-----------------------------------------------------------------------------\n', file=log)
