//
//  Networks.hpp
//  MarketGen
//
//  Created by Ved Sirdeshmukh on 24/05/20.
//  Copyright © 2020 Ved Sirdeshmukh. All rights reserved.
//

#ifndef Networks_hpp
#define Networks_hpp

#include <torch/torch.h>

class Gen : public torch::nn::Module {
 public:
     Gen(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t output_dim);
     torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::LSTM lstm;
    torch::nn::Linear fc;
};


class Discriminator : public torch::nn::Module{
public:
    Discriminator(int seq);
    torch::nn::Sequential get_model();
private:
    torch::nn::Sequential model;
};

#endif /* Networks_hpp */
