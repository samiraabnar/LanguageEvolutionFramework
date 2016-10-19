import theano
import theano.tensor as T

class ImageCaptionGenerator(object):
    def __init__(self,embedding_sapce_dim, words_dim):
        self.embedding_space_dim = embedding_sapce_dim
        self.words_dim = words_dim



    def init_weights(self):

    def define_network(self):
        def forward_step(x_t, prev_state, prev_content, prev_state_2, prev_content_2):
            input_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_input), x_t) + T.dot(self.W_input, prev_state))
            forget_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_forget), x_t) + T.dot(self.W_forget, prev_state))
            output_gate = T.nnet.hard_sigmoid(
                T.dot((self.U_output), x_t) + T.dot(self.W_output, prev_state))

            stabilized_input = T.tanh(T.dot((self.U), x_t) + T.dot(self.W, prev_state))
            c = forget_gate * prev_content + input_gate * stabilized_input
            s = output_gate * T.tanh(c)

            input_gate2 = T.nnet.hard_sigmoid(
                T.dot((self.U_input_2), s) + T.dot(self.W_input_2, prev_state_2))
            forget_gate2 = T.nnet.hard_sigmoid(
                T.dot((self.U_forget_2), s) + T.dot(self.W_forget_2, prev_state_2))
            output_gate2 = T.nnet.hard_sigmoid(
                T.dot((self.U_output_2), s) + T.dot(self.W_output_2, prev_state_2))

            stabilized_input2 = T.tanh(T.dot((self.U_2), s) + T.dot(self.W_2, prev_state_2))
            c2 = forget_gate2 * prev_content_2 + input_gate2 * stabilized_input2
            s2 = output_gate2 * T.tanh(c2)

            o = T.nnet.softmax(T.dot(self.O_w, s2))[0]

            return [o, s, c, s2, c2, input_gate, forget_gate, output_gate], theano.scan_module.until(T.eq(T.argmax(o),27))