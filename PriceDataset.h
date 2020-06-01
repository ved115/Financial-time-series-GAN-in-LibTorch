//
//  PriceDataset.h
//  MarketGen
//
//  Created by Ved Sirdeshmukh on 23/05/20.
//  Copyright © 2020 Ved Sirdeshmukh. All rights reserved.
//

#ifndef PriceDataset_h
#define PriceDataset_h

std::vector<torch::Tensor> process_prices(std::vector<float> prices, int dim);
double unnormalize(double p);

class PriceDataset : public torch::data::Dataset<PriceDataset>{
private:
    std::vector<torch::Tensor> prices;
public:
    PriceDataset(std::vector<float> list_prices, int dim) {
       prices = process_prices(list_prices, dim);
    };
    
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override {
        torch::Tensor sample_price = prices.at(index);
        torch::Tensor sample_label = torch::full(1, 1);
        return {sample_price.clone(), sample_label.clone()};
    };
    
    // Return the length of data
    torch::optional<size_t> size() const override {
      return prices.size();
    };
};


#endif /* PriceDataset_h */
