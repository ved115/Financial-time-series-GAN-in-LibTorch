//
//  Networks.cpp
//  MarketGen
//
//  Created by Ved Sirdeshmukh on 24/05/20.
//  Copyright © 2020 Ved Sirdeshmukh. All rights reserved.
//

#include "Networks.hpp"

#include <torch/torch.h>
#include <tuple>


int64_t nl;
int64_t hd;
int64_t od;

 Gen::Gen(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t output_dim)
: lstm(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers).batch_first(true).dropout(0.4)),
      fc(hidden_size, output_dim){
          
    register_module("lstm", lstm);
    register_module("fc", fc);
    nl = num_layers;
    hd = hidden_size;
    od = output_dim;
}

 torch::Tensor Gen::forward(torch::Tensor x) {
    
    torch::Tensor output;
    torch::Tensor recurrent_features;
    std::tuple<torch::Tensor, torch::Tensor> state;
    
    auto batch_size = x.size(0);
    auto seq_length = x.size(1);
    
    torch::Tensor h_0 = torch::zeros({nl, batch_size, hd});
    torch::Tensor c_0 = torch::zeros({nl, batch_size, hd});
    
    std::tuple<torch::Tensor, torch::Tensor> hx = std::make_tuple(h_0,c_0);
    
    
    std::tie(recurrent_features, state) = lstm->forward(x.to(torch::kFloat32), hx);
    output = fc->forward(recurrent_features.contiguous().reshape({batch_size*seq_length, hd}));
    output = torch::tanh(output);
    output = output.reshape({batch_size,seq_length, 1});
    return output;
    
}


Discriminator::Discriminator(int seq){
    model = torch::nn::Sequential(
        torch::nn::Conv1d(torch::nn::Conv1dOptions(seq, 32, 1).stride(2)),
        torch::nn::Functional(torch::leaky_relu, 0.01),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(32, 64, 1).stride(2)),
        torch::nn::Functional(torch::leaky_relu, 0.01),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(64).eps(0.00001).momentum(0.9)),
        torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1).stride(2)),
        torch::nn::Functional(torch::leaky_relu, 0.01),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(128).eps(0.00001).momentum(0.9)),
        torch::nn::Flatten(),
        torch::nn::Linear(128,256),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256).eps(0.00001).momentum(0.9)),
        torch::nn::Functional(torch::leaky_relu, 0.01),
        torch::nn::Linear(256,512),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(512,1),
        torch::nn::Functional(torch::sigmoid)
    );
    
}

torch::nn::Sequential Discriminator::get_model(){
    return model;
}

