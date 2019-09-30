import numpy as np
import pandas as pd
from math import exp
import copy
from sklearn import preprocessing

# Feed Forward helper methods
def sigmoid(value):
    if value < 0:
        return 1 - 1 / (1 + exp(value))
    else:
        return 1.0/(1+exp(value * (-1)))

def sigma(matrix_weight, matrix_input, bias=0):
    # Prereq: len(arr_weight) = len(arr_input)
    return matrix_weight.dot(matrix_input.transpose()) + bias

# hidden_layer = int (number of hidden layers)
# nb_nodes = arr[int] (number of nodes per hidden layer)
# len_input_matrix = int (number of features)
# Output: List of Matrixes
# Method: He initialization
# Link: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
def initialize_weights(hidden_layer, nb_nodes, len_input_matrix):
    arr_weight_this_batch = list()
    for i in range(hidden_layer):
        if i==0:
            nb_nodes_prev = len_input_matrix
        else:
            nb_nodes_prev = nb_nodes[i-1]
        weight_matrix = np.random.randn(nb_nodes[i], nb_nodes_prev) * np.sqrt(2/(nb_nodes_prev+nb_nodes[i]))
        arr_weight_this_batch.append(weight_matrix)
        
    return arr_weight_this_batch

def error(feed_forward_output, target_output):
    return ((target_output-feed_forward_output)**2)

def err_yes(value):
    return (value > 0.15) # 15% fault tolerance

# See function: initialize_errors
def initialize_sigma(hidden_layer, nb_nodes):
    list_sigma = list()
    for i in range(hidden_layer):
        arr_sigma = np.zeros(nb_nodes[i])
        list_sigma.append(arr_sigma)
    
    return list_sigma
    
# Backpropagation and Update Weight helper methods

# hidden_layer = int (number of hidden layers)
# nb_nodes = arr[int] (number of nodes per hidden layer)
# Output: List of Matrixes
def initialize_errors(hidden_layer, nb_nodes):
    arr_neuron_errors = list()
    for i in range(hidden_layer):
        arr_error = np.empty(nb_nodes[i])
        arr_neuron_errors.append(arr_error)
    
    return arr_neuron_errors

def propagate_error_output_layer(feed_forward_output, target_output):
    return feed_forward_output*(1-feed_forward_output)*(target_output-feed_forward_output)

def propagate_error_hidden_layer_neuron(sigma_output, error_contribution):
    return sigmoid(sigma_output) * (1 - sigmoid(sigma_output)) * error_contribution

# error = neuron's error
def update_weight_neuron(weight_prev_prev, weight_prev, learning_rate, momentum, error, input_neuron):
    # weight_prev_prev = previous of weight_prev
    return weight_prev + weight_prev_prev * learning_rate + momentum*error*input_neuron
    
# input_matrix = matrix[float] (data) (asumsi, kolom terakhir adalah hasil klasifikasi)
# hidden_layers = int (number of hidden layers)
# nb_nodes = arr[int] (number of nodes per hidden layer)
# nu = float (momentum)
# alfa = float (learning rate)
# epoch = int (number of training loops)
# batch_size = int (mini-batch)
# output = FFNN prediction model (list of matrix)
def mini_batch_gradient_descent(input_matrix, hidden_layer, nb_nodes, nu, alfa, epoch, batch_size=1):
    
    #transpose-slicing, memisah input dan label
    col_width = input_matrix.shape[1]
    input_col_width = col_width - 1
    input_data = (input_matrix.transpose()[0:input_col_width]).transpose()
    label_data = (input_matrix.transpose()[input_col_width:col_width]).transpose()
    
    hidden_layer += 1
    nb_nodes = np.append(nb_nodes, [1])
    arr_neuron_errors = initialize_errors(hidden_layer, nb_nodes)
    all_sigma_values = initialize_sigma(hidden_layer, nb_nodes)
    arr_weight_this_batch = initialize_weights(hidden_layer, nb_nodes, input_col_width)
    
    for no_epoch in range(epoch):
        arr_weight_prev_batch = copy.deepcopy(arr_weight_this_batch) # tracking previous state of weights
        
        batch_count = 0
        error_value = 0
        
        for no_input_data in range(len(input_data)):
            # Feed Forward
            one_sigma_values = list()
            for no_hidden_layer in range(hidden_layer):
                if no_hidden_layer == 0:
                    one_sigma_values.append(sigma(arr_weight_this_batch[no_hidden_layer], input_data[no_input_data]))
                else:
                    one_sigma_values.append(sigma(arr_weight_this_batch[no_hidden_layer], one_sigma_values[no_hidden_layer-1]))
                
                for no_rows in range(len(one_sigma_values[no_hidden_layer])):
                    output_i = sigmoid(one_sigma_values[no_hidden_layer][no_rows])
                    one_sigma_values[no_hidden_layer][no_rows] = output_i
                    all_sigma_values[no_hidden_layer][no_rows] += output_i
            #Result of sigma will be array with 1 element only, so it's safe to select like this
            error_value += error(one_sigma_values[hidden_layer - 1][0], label_data[no_input_data])[0]
            
            batch_count += 1
            
            if (batch_count == batch_size):
                error_value /= batch_size
                for no_hidden_layer in range(hidden_layer):
                    for no_neuron in range(len(all_sigma_values[no_hidden_layer])):
                        all_sigma_values[no_hidden_layer][no_neuron] /= batch_size
                
                if (err_yes(error_value)):
                    # Back Propagation
                    output_error = propagate_error_output_layer(all_sigma_values[hidden_layer-1][0], label_data[no_input_data])
                    arr_neuron_errors[hidden_layer - 1][0] = output_error

                    for no_hidden_layer in range(hidden_layer-2, -1, -1):
                        for neuron in range(nb_nodes[no_hidden_layer]):
                            # pencarian error_contribution
                            error_contribution = 0
                            for output_neuron in range(nb_nodes[no_hidden_layer+1]):
                                error_contribution += arr_weight_this_batch[no_hidden_layer + 1][output_neuron][neuron] * arr_neuron_errors[no_hidden_layer + 1][output_neuron]

                            arr_neuron_errors[no_hidden_layer][neuron] = propagate_error_hidden_layer_neuron(all_sigma_values[no_hidden_layer][neuron], error_contribution)

                    # Update Weights
                    for no_hidden_layer in range(1, hidden_layer):
                        for neuron in range(nb_nodes[no_hidden_layer]):
                            for weight in range(len(arr_weight_this_batch[no_hidden_layer][neuron])):
                                arr_weight_this_batch[no_hidden_layer][neuron][weight] = update_weight_neuron(arr_weight_prev_batch[no_hidden_layer][neuron][weight], arr_weight_this_batch[no_hidden_layer][neuron][weight], nu, alfa, arr_neuron_errors[no_hidden_layer][neuron], all_sigma_values[no_hidden_layer-1][weight])
                    #khusus hidden layer pertama, masukan dari input data
                    for neuron in range(nb_nodes[0]):
                        for weight in range(input_col_width):
                            arr_weight_this_batch[0][neuron][weight] = update_weight_neuron(arr_weight_prev_batch[0][neuron][weight], arr_weight_this_batch[0][neuron][weight], nu, alfa, arr_neuron_errors[0][neuron], input_data[no_input_data][weight])
            
            all_sigma_values = initialize_sigma(hidden_layer, nb_nodes)
            error_value = 0
            batch_count = 0
            
    return arr_weight_this_batch

# predictor specifically for dataset that is classified into only 2 classes
def predict_2classes(model, arr_features, label):
    all_sigma_values = list()
    for no_hidden_layer in range(len(model)):
        if (no_hidden_layer == 0):
            all_sigma_values.append(sigma(model[no_hidden_layer], arr_features))
        else:
            all_sigma_values.append(sigma(model[no_hidden_layer], all_sigma_values[no_hidden_layer-1]))
        for no_rows in range(len(all_sigma_values[no_hidden_layer])):
            all_sigma_values[no_hidden_layer][no_rows] = sigmoid(all_sigma_values[no_hidden_layer][no_rows])
    
    error_value = error(all_sigma_values[len(model) - 1][0], label)[0]
    return (error_value < 0.5) #scaling : (0, 1)
    
    
def accuracy(model, input_matrix):
    #transpose-slicing, memisah input dan label
    col_width = input_matrix.shape[1]
    input_col_width = col_width - 1
    input_data = (input_matrix.transpose()[0:input_col_width]).transpose()
    label_data = (input_matrix.transpose()[input_col_width:col_width]).transpose()
    
    true_count = 0
    false_count = 0
    for no_input_data in range(len(input_data)):
        if (predict_2classes(model, input_data[no_input_data], label_data[no_input_data])):
            true_count += 1
        else:
            false_count += 1
    return true_count / (true_count + false_count) * 100
    
# dataset load
csv_string = input("Input .csv filename: ")
try:
    df = pd.read_csv(csv_string)
except:
    print("File not found.")
    ###quit()
print("File loaded successfuly.")

# dataset preprocess

def preprocess_dataframe(df):
    # transform non-numeric data to numeric data
    types = df.dtypes
    labels = df.columns.values # because pandas select columns using column names
    def transform_to_numeric(matrix_data):
        for i in range(matrix_data.shape[1]):
            type_i = types[i]
            if (type_i == object):
                values = matrix_data[labels[i]].unique()
                dict_i = dict(zip(values, range(len(values)))) # transform every unique object/string into numbers
                matrix_data = matrix_data.replace({labels[i]:dict_i})
            elif (type_i == bool):
                matrix_data[labels[i]] = matrix_data[labels[i]].astype(int)
        return matrix_data

    newdf = transform_to_numeric(df)

    # scaling
    def scale_data(matrix_data, min_val, max_val):

        def scaling(value):
            return (value - minValue)*(max_val - min_val)/(maxValue - minValue) + min_val

        for x in range(matrix_data.shape[1]):
            minValue = matrix_data[labels[x]].min()
            maxValue = matrix_data[labels[x]].max()
            matrix_data[labels[x]] = matrix_data[labels[x]].apply(scaling)
        return matrix_data

    data_matrix = scale_data(newdf, 0, 1)
    data_matrix = data_matrix.to_numpy() #convert pandas dataframe to numpy array

    return data_matrix

def split_train_test(matrix_data, test_portion):
    total_data = len(matrix_data)
    total_data_for_test = int(round(test_portion * total_data, 0))
    total_data_for_train = total_data - total_data_for_test
    return(matrix_data[0:total_data_for_train], matrix_data[total_data_for_train:total_data])
    
# input and main program

while True:
    hidden_layers = int(input("Input number of hidden layers: "))
    if (hidden_layers <= 10 and hidden_layers >= 0):
        break
    else:
        print("# of hidden layers must be a positive integer and no more than 10.")

nb_nodes = np.empty(hidden_layers)
for i in range(hidden_layers):
    while True:
        nb_nodes[i] = int(input("Input number of nodes for hidden layer %d : " % i))
        if (nb_nodes[i] > 0):
            break
        else:
            print("# of nodes must be a positive integer.")
    
    
while True:
    momentum = float(input("Input momentum: "))
    if (momentum <= 1 and momentum >= 0):
        break
    else:
        print("Momentum must be between 0 and 1.")

while True:
    learning_rate = float(input("Input learning rate: "))
    if (learning_rate <= 1 and learning_rate >= 0):
        break
    else:
        print("Learning rate must be between 0 and 1.")

while True:
    epoch = int(input("Input epoch: "))
    if (epoch > 0):
        break
    else:
        print("Epoch must be a positive integer.")

while True:
    batch_size = int(input("Input the batch size: "))
    if (batch_size > 0):
        break
    else:
        print("Batch size must be a positive integer.")

while True:
    test_size = float(input("Input the test size: "))
    if (test_size > 0 and test_size < 1):
        break
    else:
        print("Test size must be between 0 and 1.")

data_matrix = preprocess_dataframe(df)
train_matrix, test_matrix = split_train_test(data_matrix, test_size)

nb_nodes = nb_nodes.astype(int) #diperlukan karena dianggap float dalam fungsi randn jika tak diubah
custom_model = mini_batch_gradient_descent(train_matrix, hidden_layers, nb_nodes, momentum, learning_rate, epoch, batch_size)
print("Accuracy: ", accuracy(custom_model, test_matrix))
