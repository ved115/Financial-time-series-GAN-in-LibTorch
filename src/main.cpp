//
//  main.cpp
//  MarketGen
//
//  Created by Ved Sirdeshmukh on 23/05/20.
//  Copyright © 2020 Ved Sirdeshmukh. All rights reserved.
//

#include <iomanip>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <tuple>
#include <iterator>
#include <iostream>
#include <torch/torch.h>
#include "PriceDataset.hpp"
#include "Networks.hpp"

using namespace torch::indexing;


int main(int argc, char** argv){
    
    // Device Check
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
    
    //Hyperparameters
    const int64_t batch_size = 16;
    const int64_t hidden_size = 256;
    const int64_t num_layers = 2;
    const int64_t out_dim = 1;
    const int64_t latent_size = 100;
    const int64_t timesteps = 32;
    const double learning_rate = 0.0002;
    const size_t num_epochs = 50;

    //Obtaining data line by line from CSV
    std::string line;
    std::ifstream classFile("EOD-AAPL.csv");
    std::vector<float> PriceData;
    
    while (getline(classFile, line,',')){
        PriceData.push_back(std::stof(line));
    }
    
    //Defining Custom Dataset class and Dataloader
    auto custom_dataset = PriceDataset(PriceData,timesteps).map(torch::data::transforms::Stack<>());
    
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                                                std::move(custom_dataset), torch::data::DataLoaderOptions(batch_size).drop_last(true));
    
    
    //Defining Networks
    
    Gen G = Gen(latent_size, hidden_size, num_layers, out_dim);

    Discriminator Dis = Discriminator(timesteps);
    torch::nn::Sequential D = Dis.get_model();
    
    D->to(device);
    G.to(device);
    
    //Defining Optimizers
    torch::optim::SGD d_optimizer(D->parameters(), torch::optim::SGDOptions(learning_rate));
    torch::optim::Adam g_optimizer(G.parameters(), torch::optim::AdamOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);
    
    
    std::cout << "Training...\n";
    
    torch::Tensor prices;
    torch::Tensor fake_prices;
    int batch_index = 0;
    
    for(size_t epoch = 0; epoch != num_epochs; ++epoch) {
    
        for (auto& batch : *dataloader) {
            
          //DISCRIMINATOR TRAINING
            
            //Train with Real Data
            D->zero_grad();
            torch::Tensor label_real = torch::full({batch_size, 1}, 1).to(device);
            torch::Tensor label_fake = torch::full({batch_size, 1}, 0).to(device);
            
            prices = batch.data.to(device);
            auto output = D->forward(prices.to(torch::kFloat32));
            auto d_loss_real = torch::nn::functional::mse_loss(output, label_real);
            auto real_score = output.mean().item<double>();
            
            // Backward pass and optimize
            D->zero_grad();
            d_loss_real.backward();
            d_optimizer.step();

            
            //Train with Fake Data
            torch::Tensor noise = torch::randn({batch_size, timesteps, latent_size}).to(device);
            fake_prices = G.forward(noise);
            output = D->forward(fake_prices);
            auto d_loss_fake = torch::nn::functional::mse_loss(output, label_fake);
            auto fake_score = output.mean().item<double>();

            auto d_loss = d_loss_real + d_loss_fake;
            
            // Backward pass and optimize
            D->zero_grad();
            d_loss_fake.backward();
            d_optimizer.step();

            //GENERATOR TRAINING
            
            // Compute loss with fake images
            noise = torch::randn({batch_size, timesteps, latent_size}).to(device);
            fake_prices = G.forward(noise);
            output = D->forward(fake_prices);
            
            auto g_loss = torch::nn::functional::mse_loss(output, label_real);
            
            // Backward pass and optimize
            G.zero_grad();
            g_loss.backward();
            g_optimizer.step();
            
            
            //Print status periodically
            if ((batch_index+1) % 20 == 0 ) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], d_loss: " << d_loss.item<double>() << ", g_loss: "
                    << g_loss.item<double>() << ", D(x): " << real_score
                    << ", D(G(z)): " << fake_score << "\n";
            }

            batch_index = batch_index+1;
            
        }
        

    }
    
    //Output fake prices
    torch::Tensor fake_sample;
    std::vector<double> fake_price_normal;
    torch::Tensor noise = torch::randn({1,100,latent_size});
    fake_sample = G.forward(noise);
    fake_sample = fake_sample.view({1,-1});
    
    //Un-normalize Prices
    for(int i = 0; i<fake_sample.size(1);i++){
        double num = fake_sample[0][i].item<double>();
        num = unnormalize(num);
        fake_price_normal.push_back(num);
    }
    
    //Save data to a file
    std::ofstream myfile;
    myfile.open("fake_prices.csv");
    
    for(int i = 0; i<fake_price_normal.size();i++ ){
        if(i == 0)
            myfile<<"Fake Prices"<<std::endl;
        myfile << fake_price_normal.at(i)<<std::endl;
    }
    
    myfile.close();
    
    //Print the prices
    std::cout << "Training finished!\n";
    std::cout << "Fake Prices are : "<< fake_price_normal;
    
}

