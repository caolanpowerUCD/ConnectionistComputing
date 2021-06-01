import numpy as np
from MLP import MLP

log = open("SIN_test50Hidden.txt", "w")
print("Q2 SIN_test\n", file=log)


def SIN(max_epochs, learning_rate, NH):
    inputs = []
    outputs = []

    print('Epochs = ' + str(max_epochs))
    print('Epochs = ' + str(max_epochs), file=log)

    print('Learning rate = ' + str(learning_rate))
    print('Learning rate = ' + str(learning_rate), file=log)

    print('Hidden units = ' + str(NH))
    print('Hidden units = ' + str(NH) + '\n\n', file=log)

    for i in range(500):
        vector = list(np.random.uniform(-1.0, 1.0, 4))
        vector = [float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])]

        inputs.append(vector)

    inputs = np.array(inputs)

    for i in range(500):
        outputs.append(np.sin([inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]]))

    num_inputs = 4
    num_outputs = 1
    NN = MLP(num_inputs, NH, num_outputs)

    NN.randomise()

    # training
    for i in range(max_epochs):
        NN.forward(inputs[:400], 'tanh')
        error = NN.backward(inputs[:400], outputs[:400], 'tanh')
        NN.updateWeights(learning_rate)

        if (i + 1) == 100 or (i + 1) == 1000 or (i + 1) == 10000 or (i + 1) == 100000 or (i + 1) == 1000000:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=log)

    # testing
    diff = 0
    for i in range(400, len(inputs)):
        NN.forward(inputs[i], 'tanh')
        diff = diff + np.abs(outputs[i][0] - NN.O[0])

    print(diff)
    accuracy = 1-(diff/100)
    print('\nAccuracy = ' + str(accuracy), file=log)


epochs = [100000]
learning_rate = [0.1, 0.01, 0.001, 0.0001]
num_hidden = 50

for i in range(len(epochs)):
    for j in range(len(learning_rate)):
        SIN(epochs[i], learning_rate[j], num_hidden)
        print('\n-----------------------------------------------------------------------------------\n', file=log)
