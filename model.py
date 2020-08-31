import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
       
        super(DecoderRNN, self).__init__()
        
        # SSSSembedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # Setting dropout
        dropout = 0 if num_layers == 1 else 0.2
        
        # LSTM network
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # FC layer (hidden_size -> vocab_size)
        self.hid2pred = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        
        # create embedded word vectors for each word in a sentence
        embeddings = self.word_embeddings(captions[:, :-1])
        
        # Concatenate the features and caption inputs
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # Run the inputs (captions + CNN embessing) through the LSTM
        lstm_out, _ = self.lstm(inputs, None)
        
        # Use linear layer to "reshape" dims to vocab_size and make preds
        out = self.hid2pred(lstm_out)
                
        return out
        

    def sample(self, inputs, states=None, max_len=20):
        
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "   
        words = []
        embeddings = inputs
        hidden_state = None
        idx = 0
        
        while (idx != 1) and len(words) < max_len:
            
            lstm_out, hidden_state = self.lstm(embeddings, hidden_state)
                                    
            out = self.hid2pred(lstm_out)
                                    
            _, idx = torch.topk(out.squeeze(), k=1)
            
            words.append(idx.cpu().item())

            embeddings = self.word_embeddings(idx).unsqueeze(1)
        
        
        return words