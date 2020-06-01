//
//  PriceDataset.cpp
//  MarketGen
//
//  Created by Ved Sirdeshmukh on 23/05/20.
//  Copyright © 2020 Ved Sirdeshmukh. All rights reserved.
//

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include "PriceDataset.h"

float max_p;
float min_p;

std::vector<torch::Tensor> process_prices(std::vector<float> prices, int dim){
    
    std::vector<torch::Tensor> states;
    std::vector<float> batchV;
    int seq_length = dim;
    
    //Max min for normalizing
    max_p = *max_element (prices.begin(), prices.end());
    min_p = *min_element (prices.begin(), prices.end());
    
    //Take prices into vector
    for(int i = 0; i<prices.size(); i++){
        float p = prices.at(i);
        
        //Normalize each price
        p = (2 * (p - min_p))/(max_p - min_p) - 1;
        batchV.push_back(p);
        
        //Convert a set of prices equal to the sequence length into a tensor and store it in a vector
        if((i+1) % seq_length == 0){
            auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor batchT = torch::from_blob(batchV.data(), {seq_length,1},opts).to(torch::kFloat64);
            states.push_back(batchT);
            batchV.clear();
        }
        
    }
    return states;
}

//Unnormalize function
double unnormalize(double p){
    double u_p = 0.5*((p*max_p)-(p*min_p) + max_p + min_p);
    return u_p;
}
