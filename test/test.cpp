#include "../include/GANN.hpp"
#include "../dependencies/Neural-Network/dependencies/cpp-easy-file-stream/include/fs.hpp"
#include <vector>

using namespace GANN;

void readDataset(std::string fileName, std::vector<std::vector<double>> &inputs,
                 std::vector<std::vector<double>> &outputs, int numberOfCollums,
                 int outputCollum) {
  FileStream fs(fileName);
  std::string word;
  std::map<std::string, double> classes;
  word = fs.getDelimiter(',');
  while (word != "") {
    std::vector<double> input;
    for (int i = 0; i < numberOfCollums; i++) {
      if (i == outputCollum) {
        if (classes.find(word) == classes.end()) {
          classes[word] = classes.size();
        }
        outputs.push_back({classes[word]});
        word = fs.getDelimiter(',');

        continue;
      }
      input.push_back(word == "?" ? 0.0 : std::stod(word));
      word = fs.getDelimiter(',');
    }
    inputs.push_back(input);
  }

  for (auto &x : outputs) {
    int classification = x[0];
    x.clear();
    for (int i = 0; i < classes.size(); i++) {
      if (i == classification) {
        x.push_back(1);
      } else {
        x.push_back(0);
      }
    }
  }
}

void normalizeInputs(std::vector<std::vector<double>> &inputs) {
  std::vector<double> maxValues(inputs[0].size(), DBL_MIN);
  std::vector<double> minValues(inputs[0].size(), DBL_MAX);
  for (auto &x : inputs) {
    for (int i = 0; i < x.size(); i++) {
      maxValues[i] = std::max(maxValues[i], x[i]);
      minValues[i] = std::min(minValues[i], x[i]);
    }
  }
  for (auto &x : inputs) {
    for (int i = 0; i < x.size(); i++) {
      x[i] = (x[i] - minValues[i]) / (maxValues[i] - minValues[i]);
    }
  }
}

int main() {
  // std::vector<int> topology = {2,4,2};
  // std::vector<std::vector<double>> inputValues = {{0,0},{0,1},{1,0},{1,1}};
  // std::vector<std::vector<double>> outputValues = {{0,1},{1,0},{1,0},{0,1}};

  std::vector<std::vector<double>> irisInputs, irisOutputs;
  readDataset("iris.csv", irisInputs, irisOutputs, 5, 4);
  normalizeInputs(irisInputs);

  std::vector<std::vector<double>> wineInputs, wineOutputs;
  readDataset("wine.csv", wineInputs, wineOutputs, 14, 0);
  normalizeInputs(wineInputs);

  float targetAcc = 0.90;

  int inputSize = wineInputs[0].size();
  int outputSize = wineOutputs[0].size();
  YAGANN gann({inputSize, 10, outputSize}, wineInputs, wineOutputs);
  for (int i = 0; i < 2000; i++) {
    gann.run(10);
    std::cout << "\nGeneration: " << i;
    std::cout << "\tFitness: ";
    float fitness = gann.getBestIndividual().getFitness();
    std::cout << fitness;
    if (fitness >= targetAcc) {
      std::cout << "\nReached target fitness\n";
      break;
    }
    // std::cout << "\tGenome: ";
    // std::cout << gann.getBestIndividual().getGenotype();
  }
  std::cout << "\n";
  auto nn = gann.getBestIndividual().getPhenotype();
  nn.setInputs(wineInputs);
  nn.setOutputs(wineOutputs);
  nn.testWithOutput();
  return 0;
}
