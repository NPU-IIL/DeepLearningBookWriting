import numpy as np   
from scipy.special import expit as sigmoid

def forget_gate(x, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, prev_cell_state):
    forget_hidden  = np.dot(Weights_hf, h) + Bias_hf
    forget_eventx  = np.dot(Weights_xf, x) + Bias_xf
    return np.multiply( sigmoid(forget_hidden + forget_eventx), prev_cell_state )

def input_gate(x, h, Weights_hi, Bias_hi, Weights_xi, Bias_xi, Weights_hl, Bias_hl, Weights_xl, Bias_xl):
    ignore_hidden  = np.dot(Weights_hi, h) + Bias_hi
    ignore_eventx  = np.dot(Weights_xi, x) + Bias_xi
    learn_hidden   = np.dot(Weights_hl, h) + Bias_hl
    learn_eventx   = np.dot(Weights_xl, x) + Bias_xl
    return np.multiply( sigmoid(ignore_eventx + ignore_hidden), np.tanh(learn_eventx + learn_hidden) )

def cell_state(forget_gate_output, input_gate_output):
    return forget_gate_output + input_gate_output

def output_gate(x, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, cell_state):
    out_hidden = np.dot(Weights_ho, h) + Bias_ho
    out_eventx = np.dot(Weights_xo, x) + Bias_xo
    return np.multiply( sigmoid(out_eventx + out_hidden), np.tanh(cell_state) )

input_size  = 2
hidden_dim  = 3
output_size = 1

def model_output(lstm_output, fc_Weight, fc_Bias):
    return np.dot(fc_Weight, lstm_output) + fc_Bias

