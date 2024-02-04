#pragma once

#include "../dependencies/Neural-Network/include/ClassificationNN.hpp"
#include "../dependencies/Neural-Network/include/NeuralNetwork.hpp"
#include "../dependencies/generic-ga/include/GA.hpp"
#include "../dependencies/generic-ga/include/Helper.hpp"
#include "../dependencies/generic-ga/include/Individual.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define DBL_MAX 1.79769e+308
#define DBL_MIN 2.22507e-308

namespace GANN {
class NNStrategy : public GenotypePhenotypeStrategy<ClassificationNN> {
private:
  std::vector<int> topology;

public:
  NNStrategy(const std::vector<int> &t) { topology = t; }
  ClassificationNN
  genotype2phenotype(const std::vector<bool> &genotype) override {
    ClassificationNN nn(topology, "sigmoid");
    auto layers = nn.getLayers();
    int index = 0;
    for (auto &layer : layers) {
      for (auto &neuron : layer.neurons) {
        for (auto &weight : neuron.weights) {
          int signal = genotype[index] == true ? 1 : -1;
          index++;
          std::vector<bool> integerPartVec(genotype.begin() + index,
                                           genotype.begin() + index + 4);
          int integerPart = Helper::gray2decimal(integerPartVec);
          index += 4;
          std::vector<bool> decimalPartVec(genotype.begin() + index,
                                           genotype.begin() + index + 15);
          float decimalPart = Helper::gray2float(decimalPartVec);
          index += 15;
          weight = signal * (integerPart + decimalPart);
        }
        int signal = genotype[index] == true ? 1 : -1;
        index++;
        std::vector<bool> integerPartVec(genotype.begin() + index,
                                         genotype.begin() + index + 4);
        int integerPart = Helper::gray2decimal(integerPartVec);
        index += 4;
        std::vector<bool> decimalPartVec(genotype.begin() + index,
                                         genotype.begin() + index + 15);
        float decimalPart = Helper::gray2float(decimalPartVec);
        index += 15;
        neuron.bias = signal * (integerPart + decimalPart);
      }
    }

    return nn;
  }

  std::vector<bool>
  phenotype2genotype(const ClassificationNN &phenotype) override {
    // Neural Network data to encode:
    // 1. Weights
    // 2. Biases
    // Each weight and bias is encoded as a 1 bit signal + 4 bit integer part +
    // 15 bit decimal part

    auto layers = phenotype.getLayers();
    std::vector<bool> genome{};
    for (auto &layer : layers) {
      for (auto &neuron : layer.neurons) {
        for (auto &weight : neuron.weights) {
          genome.push_back(weight > 0 ? true : false);
          auto decimal = Helper::decimal2gray((int)weight, 4);
          genome.insert(genome.end(), decimal.begin(), decimal.end());
          auto fraction = Helper::float2gray(weight - (int)weight, 15);
          genome.insert(genome.end(), fraction.begin(), fraction.end());
        }
        genome.push_back(neuron.bias > 0 ? true : false);
        auto decimal = Helper::decimal2gray((int)neuron.bias, 4);
        genome.insert(genome.end(), decimal.begin(), decimal.end());
        auto fraction = Helper::float2gray(neuron.bias - (int)neuron.bias, 15);
        genome.insert(genome.end(), fraction.begin(), fraction.end());
      }
    }
    return genome;
  }
};
class YagannGeneratorStrategy
    : public IIndividualGenerationStrategy<Individual<ClassificationNN>> {
  std::vector<int> topology;
  std::vector<std::vector<double>> inputValues;
  std::vector<std::vector<double>> outputValues;

public:
  YagannGeneratorStrategy(
      const std::vector<int> &topology,
      const std::vector<std::vector<double>> &inputValues,
      const std::vector<std::vector<double>> &outputValues) {
    this->topology = topology;
    this->inputValues = inputValues;
    this->outputValues = outputValues;
  }
  Individual<ClassificationNN> generateIndividual() override {
    return Individual<ClassificationNN>(
        ClassificationNN(topology, inputValues, outputValues, "tanh"));
  }
};

class YagannFitnessStrategy
    : public IFitnessCalculationStrategy<Individual<ClassificationNN>> {

  std::vector<std::vector<double>> inputValues;
  std::vector<std::vector<double>> outputValues;

public:
  YagannFitnessStrategy(const std::vector<std::vector<double>> &inputValues,
                        const std::vector<std::vector<double>> &outputValues) {
    this->inputValues = inputValues;
    this->outputValues = outputValues;
  }
  float calculateFitness(const Individual<ClassificationNN> &i) override {
    ClassificationNN nn = i.getPhenotype();
    nn.setInputs(inputValues);
    nn.setOutputs(outputValues);
    return nn.test();
  }
};

class YagannViabilityStrategy
    : public IViabilityStrategy<Individual<ClassificationNN>> {
public:
  bool isViable(const Individual<ClassificationNN> &i) override { return true; }
};

class YAGANN {
public:
  std::vector<int> topology;
  std::vector<std::vector<double>> inputValues;
  std::vector<std::vector<double>> outputValues;
  std::unique_ptr<GA<Individual<ClassificationNN>>> ga;

  YAGANN(std::vector<int> t,
         const std::vector<std::vector<double>> &inputValues,
         const std::vector<std::vector<double>> &outputValues) {
    this->topology = t;
    this->inputValues = inputValues;
    this->outputValues = outputValues;
    NNStrategy *strategy = new ::GANN::NNStrategy(t);
    Individual<ClassificationNN>::setStrategy(strategy);

    ga = std::make_unique<GA<Individual<ClassificationNN>>>(
        std::make_unique<GANN::YagannFitnessStrategy>(inputValues,
                                                      outputValues),
        std::make_unique<GANN::YagannViabilityStrategy>(),
        std::make_unique<GANN::YagannGeneratorStrategy>(t, inputValues,
                                                        outputValues));
  }

  void run(int generations) { ga->run(generations); }

  Individual<ClassificationNN> getBestIndividual() {
    return ga->getBestIndividual();
  }
};
} // namespace GANN
