import paddle
import paddle.nn as nn

class Autoencoder():
    def __init__(self, input_dim, hidden_dims=(16,), output_activation='sigmoid', optimiser='adam',
                 learning_rate=0.001, loss='mse'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_activation = output_activation
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.loss = loss
        self.model = paddle.Model(AutoencoderNet())
    def train(self, X_train, y_train, epochs=5, batch_size=256, val_data=None, ):
        self.model.fit(X_train, eval_data = val_data, batch_size = batch_size, epochs = epochs,)
        
    def 

    
    
    



class AutoencoderNet(nn.Layer):
    def __init__(self, input_dim, hidden_dims=(16,)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.linears  = nn.LayerList()
        self.linears.append(nn.Linear(input_dim, hidden_dims[0]))
        self.linears.append(nn.ReLU())
        for i in range(len(self.hidden_dims)):
            self.linears.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            self.linears.append(nn.ReLU())
        self.linears[-1] = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linears(x)
        return x


