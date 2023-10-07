hidden_dim = 100     
output_dim = 80
input_weights = np.random.uniform(0, 1,  (hidden_dim, hidden_dim))
internal_state_weights = np.random.uniform(0,1, (hidden_dim, hidden_dim))
output_weights = np.random.uniform(0,1, (output_dim,hidden_dim))

input_string = [2,45,10,65]
embeddings = []
for i in range(0,T):
    x = np.random.randn(hidden_dim,1)
    embeddings.append(x)

output_mapper = {}
for index_value in output_string :
    output_mapper[index_value] = identity_matrix[index_value,:]

output_t = {}
i=0
for key,value in output_mapper.items():
    output_t[i] = value
    i += 1

def tanh_activation(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)-np.exp(-Z))

def softmax_activation(Z):
    e_x = np.exp(Z - np.max(Z))
    return e_x / e_x.sum(axis=0)


def Rnn_forward(input_embedding, input_weights, internal_state_weights, prev_memory,output_weights):
    forward_params = []
    W_frd = np.dot(internal_state_weights,prev_memory)
    U_frd = np.dot(input_weights,input_embedding)
    sum_s = W_frd + U_frd
    ht_activated = tanh_activation(sum_s)
    yt_unactivated = np.asarray(np.dot(output_weights,  tanh_activation(sum_s)))
    yt_activated = softmax_activation(yt_unactivated)
    forward_params.append([W_frd,U_frd,sum_s,yt_unactivated])
    return ht_activated,yt_activated,forward_params



