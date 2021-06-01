import numpy as np
from MLP import MLP
import csv

log = open("LETTER_test20Hidden.txt", "w")
print("Q3 LETTER TEST\n", file=log)


def letter(max_epochs, learning_rate, NH):
    data = []
    inputs = []
    outputs = []
    inputs_train = []
    outputs_train = []
    inputs_test = []
    outputs_test = []
    recieved_outputs = []

    print('Epochs = ' + str(max_epochs))
    print('Epochs = ' + str(max_epochs), file=log)

    print('Learning rate = ' + str(learning_rate))
    print('Learning rate = ' + str(learning_rate), file=log)

    print('Hidden units = ' + str(NH))
    print('Hidden units = ' + str(NH) + '\n\n', file=log)

    with open('letter-recognition.data', 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data.append(row)

    for i in range(len(data)):
        outputs.append(data[i][0])
        inputs.append(data[i][1:])

    # training set
    inputs_train = inputs[:16000]
    outputs_train = outputs[:16000]
    inputs_train = np.array(inputs_train)
    outputs_train = np.array(outputs_train)
    inputs_train = inputs_train.astype(int)

    # enumerate outputs
    real_output = np.zeros((20000, 26))
    for i in range(len(real_output)):
        if outputs[i] == 'A':
            real_output[i][0] = 1
        elif outputs[i] == 'B':
            real_output[i][1] = 1
        elif outputs[i] == 'C':
            real_output[i][2] = 1
        elif outputs[i] == 'D':
            real_output[i][3] = 1
        elif outputs[i] == 'E':
            real_output[i][4] = 1
        elif outputs[i] == 'F':
            real_output[i][5] = 1
        elif outputs[i] == 'G':
            real_output[i][6] = 1
        elif outputs[i] == 'H':
            real_output[i][7] = 1
        elif outputs[i] == 'I':
            real_output[i][8] = 1
        elif outputs[i] == 'J':
            real_output[i][9] = 1
        elif outputs[i] == 'K':
            real_output[i][10] = 1
        elif outputs[i] == 'L':
            real_output[i][11] = 1
        elif outputs[i] == 'M':
            real_output[i][12] = 1
        elif outputs[i] == 'N':
            real_output[i][13] = 1
        elif outputs[i] == 'O':
            real_output[i][14] = 1
        elif outputs[i] == 'P':
            real_output[i][15] = 1
        elif outputs[i] == 'Q':
            real_output[i][16] = 1
        elif outputs[i] == 'R':
            real_output[i][17] = 1
        elif outputs[i] == 'S':
            real_output[i][18] = 1
        elif outputs[i] == 'T':
            real_output[i][19] = 1
        elif outputs[i] == 'U':
            real_output[i][20] = 1
        elif outputs[i] == 'V':
            real_output[i][21] = 1
        elif outputs[i] == 'W':
            real_output[i][22] = 1
        elif outputs[i] == 'X':
            real_output[i][23] = 1
        elif outputs[i] == 'Y':
            real_output[i][24] = 1
        elif outputs[i] == 'Z':
            real_output[i][25] = 1

    # test set
    inputs_test = inputs[16000:]
    inputs_test = np.array(inputs_test)
    inputs_test = inputs_test.astype(int)
    outputs_test = real_output[16000:]
    outputs_train = real_output[:16000]

    num_inputs = 16
    num_outputs = 26
    NN = MLP(num_inputs, NH, num_outputs)
    NN.randomise()

    # initial test
    for i in range(4000):
        NN.forward(inputs_test[i], 'tanh')

    print("O ==============")
    print(len(NN.O))
    print(NN.O)
    print("REAL ============")
    print(len(real_output))
    print(real_output[1600:])
    initial_error = np.subtract(real_output[1600:], NN.O)
    print("INITIAL ERROR ==============")
    print(np.mean(np.abs(initial_error)))
    print('Error before training = ' + str(np.mean(np.abs(initial_error))) + '\n', file=log)

    # training
    for i in range(max_epochs):
        NN.forward(inputs_train, 'tanh')
        error = NN.backward(inputs_train, outputs_train, 'tanh')
        NN.updateWeights(learning_rate)

        if (i + 1) == 100 or (i + 1) == 1000 or (i + 1) == 10000 or (i + 1) == 100000 or (i + 1) == 1000000:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(np.mean(np.abs(error))), file=log)

    # testing
    for i in range(4000):
        NN.forward(inputs_test[i], 'tanh')

    print(type(real_output[1600:][0]))
    print(type(NN.O))

    test_error = np.subtract(real_output[1600:], NN.O)
    print("TEST ERROR =============")
    print(np.mean(np.abs(test_error)))
    print('\nError after training = ' + str(np.mean(np.abs(test_error))), file=log)


epochs = [100000]
learning_rate = [0.00005]
num_hidden = 20
for i in range(len(epochs)):
    for j in range(len(learning_rate)):
        print('--------------------------------------------------------------------------\n', file=log)
        letter(epochs[i], learning_rate[j], num_hidden)
