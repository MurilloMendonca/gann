#pragma once

#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include<chrono>
#include "Neural-Network/NeuralNetwork.hpp"
#include "generic-ga/GA.hpp"
#include "generic-ga/Individual.hpp"
#include "generic-ga/Helper.hpp"

#define DBL_MAX 1.79769e+308
#define DBL_MIN 2.22507e-308

namespace GANN{
	class NNStrategy : public GenotypePhenotypeStrategy<NeuralNetwork> {
	private:
		std::vector<int> topology;
	public:
		NNStrategy(const std::vector<int>& t) {
			topology = t;
		}
		NeuralNetwork genotype2phenotype(const std::string& genotype) override {
			NeuralNetwork nn(topology, "sigmoid");
			auto layers = nn.getLayers();
			int index = 0;
			for(auto& layer : layers){
				for(auto& neuron : layer.neurons){
					for(auto& weight : neuron.weights){
						int signal = genotype[index] == '1' ? 1 : -1;
						index++;
						int integerPart = Helper::gray2decimal(genotype.substr(index, 4));
						index += 4;
						float decimalPart = Helper::gray2float(genotype.substr(index, 15));
						index += 15;
						weight = signal * (integerPart + decimalPart);
					}
					int signal = genotype[index] == '1' ? 1 : -1;
					index++;
					int integerPart = Helper::gray2decimal(genotype.substr(index, 4));
					index += 4;
					float decimalPart = Helper::gray2float(genotype.substr(index, 15));
					index += 15;
					neuron.bias = signal * (integerPart + decimalPart);
				}
			}

			return nn;
			
		}

		std::string phenotype2genotype(const NeuralNetwork& phenotype) override {
			// Neural Network data to encode:
			// 1. Weights
			// 2. Biases
			// Each weight and bias is encoded as a 1 bit signal + 4 bit integer part + 15 bit decimal part

			auto layers = phenotype.getLayers();
			std::string genome = "";
			for(auto& layer : layers){
				for(auto& neuron : layer.neurons){
					for(auto& weight : neuron.weights){
						genome += weight > 0 ? "1" : "0";
						genome += Helper::decimal2gray((int)weight, 4);
						genome += Helper::float2gray(weight - (int)weight, 15);
					}
					genome += neuron.bias > 0 ? "1" : "0";
					genome += Helper::decimal2gray((int)neuron.bias, 4);
					genome += Helper::float2gray(neuron.bias - (int)neuron.bias, 15);
				}
			}
			return genome;


		}
	};
	void setTopology(const std::vector<int>& t);
	void setTopology(int input, int hidden, int output);
	GA<Individual<NeuralNetwork>>* createGA(std::vector<std::vector<double>> inputValues, std::vector<std::vector<double>> outputValues);
	GA<Individual<NeuralNetwork>>* createGA(std::vector<int> t,std::vector<std::vector<double>> inputValues, std::vector<std::vector<double>> outputValues);
	float run(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs);
	float runWithOutput(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs);
	class YAGANN : public GA<Individual<NeuralNetwork>>{
		public:
			std::vector<int> topology;
			std::vector<std::vector<double>> inputValues;
			std::vector<std::vector<double>> outputValues;
			YAGANN(std::vector<int> t,const std::vector<std::vector<double>>& inputValues, const std::vector<std::vector<double>>& outputValues)
				:GA<Individual<NeuralNetwork>>(100,0.3,0.01){	
				this->topology = t;
				this->inputValues = inputValues;
				this->outputValues = outputValues;
				NNStrategy* strategy = new ::GANN::NNStrategy(t);
				Individual<NeuralNetwork>::setStrategy(strategy);

				initializePopulation();
			}
			Individual<NeuralNetwork> generateValidIndividual() override{
				return Individual<NeuralNetwork>(NeuralNetwork(topology, "tanh"));
			}

			float calculateFitness(Individual<NeuralNetwork> i) override{
				NeuralNetwork nn = i.getPhenotype();
				return ::GANN::run(nn, inputValues, outputValues);
			}

			bool isViable(Individual<NeuralNetwork>) override{
				return true;
			}





	};

}
