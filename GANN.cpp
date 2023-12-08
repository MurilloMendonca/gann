#include "GANN.hpp"




namespace GANN{

	
	


	void normalizeInputs(std::vector<std::vector<double>>& inputs){
		std::vector<double> maxValues(inputs[0].size(),DBL_MIN);
		std::vector<double> minValues(inputs[0].size(),DBL_MAX);
		for(auto& x : inputs){
			for(int i = 0;i<x.size();i++){
				maxValues[i] = std::max(maxValues[i],x[i]);
				minValues[i] = std::min(minValues[i],x[i]);
			}
		}
		for(auto& x : inputs){
			for(int i = 0;i<x.size();i++){
				x[i] = (x[i]-minValues[i])/(maxValues[i]-minValues[i]);
			}
		}

	}
	void train(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, int epochs = 10000) {
		for (int i = 0; i < epochs; ++i) {
			for (size_t j = 0; j < inputs.size(); ++j) {
				nn.forward(inputs[j]);
				nn.backpropagate(outputs[j]);
			}
		}
	}

	float run(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs) {
		double totalError = 0.0;
		int errorNumber = 0;
		std::vector<int> truePositives(outputs[0].size(), 0);  // Initialize true positives for each class to 0
		std::vector<int> falsePositives(outputs[0].size(), 0); // Initialize false positives for each class to 0

		for (size_t i = 0; i < inputs.size(); ++i) {
			nn.forward(inputs[i]);
			const auto& resultLayer = nn.getLayers().back().neurons;

			double exampleError = 0.0;
			int classification = 0;
			for(size_t j = 0; j < outputs[0].size(); j++) {
				double error = outputs[i][j] - resultLayer[j].value;
				exampleError += error * error;  // squared error
				if(resultLayer[j].value > resultLayer[classification].value) {
					classification = j;
				}
			}
			int expectedClassification = 0;
			for(size_t j = 0; j < outputs[0].size(); j++) {
				if(outputs[i][j] > outputs[i][expectedClassification]) {
					expectedClassification = j;
				}
			}
			if(expectedClassification == classification) {
				truePositives[classification]++;
			} else {
				falsePositives[classification]++;
				errorNumber++;
			}
			totalError += exampleError / outputs[0].size();  // average error for this example
		}

		double mse = totalError / inputs.size();  // mean squared error over all examples
		//std::cout << "Mean Squared Error (MSE) on All Data: " << mse << std::endl;

		//std::cout<<"Wrong Predictions: "<<errorNumber<<std::endl;
		//std::cout<<"Right Predictions: "<<inputs.size()-errorNumber<<std::endl;
		//std::cout<<"Accuracy: "<<(double)(inputs.size()-errorNumber)/inputs.size()*100<<"%"<<std::endl;

		// Compute and display precision for each class
		//for (size_t i = 0; i < outputs[0].size(); i++) {
			//double precision = static_cast<double>(truePositives[i]) / (truePositives[i] + falsePositives[i]);
			//std::cout << "Precision for class " << i << ": " << precision * 100 << "%" << std::endl;
		//}

		return (double)(inputs.size()-errorNumber)/inputs.size();

	}

	float runWithOutput(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs) {
		double totalError = 0.0;
		int errorNumber = 0;
		std::vector<int> truePositives(outputs[0].size(), 0);  // Initialize true positives for each class to 0
		std::vector<int> falsePositives(outputs[0].size(), 0); // Initialize false positives for each class to 0

		for (size_t i = 0; i < inputs.size(); ++i) {
			nn.forward(inputs[i]);
			const auto& resultLayer = nn.getLayers().back().neurons;

			double exampleError = 0.0;
			int classification = 0;
			for(size_t j = 0; j < outputs[0].size(); j++) {
				double error = outputs[i][j] - resultLayer[j].value;
				exampleError += error * error;  // squared error
				if(resultLayer[j].value > resultLayer[classification].value) {
					classification = j;
				}
			}
			int expectedClassification = 0;
			for(size_t j = 0; j < outputs[0].size(); j++) {
				if(outputs[i][j] > outputs[i][expectedClassification]) {
					expectedClassification = j;
				}
			}
			if(expectedClassification == classification) {
				truePositives[classification]++;
			} else {
				falsePositives[classification]++;
				errorNumber++;
			}
			totalError += exampleError / outputs[0].size();  // average error for this example
		}

		double mse = totalError / inputs.size();  // mean squared error over all examples
		std::cout << "Mean Squared Error (MSE) on All Data: " << mse << std::endl;

		std::cout<<"Wrong Predictions: "<<errorNumber<<std::endl;
		std::cout<<"Right Predictions: "<<inputs.size()-errorNumber<<std::endl;
		std::cout<<"Accuracy: "<<(double)(inputs.size()-errorNumber)/inputs.size()*100<<"%"<<std::endl;

		// Compute and display precision for each class
		for (size_t i = 0; i < outputs[0].size(); i++) {
			double precision = static_cast<double>(truePositives[i]) / (truePositives[i] + falsePositives[i]);
			std::cout << "Precision for class " << i << ": " << precision * 100 << "%" << std::endl;
		}

		return mse;

	}


	void trainAndTest(NeuralNetwork& nn, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, int epochs = 100) {
		// Shuffle the dataset
		auto shuffled_indices = std::vector<size_t>(inputs.size());
		std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);  // Fill with 0, 1, ..., n-1
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), std::default_random_engine(seed));

		// Split data into 80% training and 20% test
		size_t trainingSize = inputs.size() * 0.8;
		std::vector<std::vector<double>> trainingInputs(trainingSize);
		std::vector<std::vector<double>> trainingOutputs(trainingSize);
		std::vector<std::vector<double>> testInputs(inputs.size() - trainingSize);
		std::vector<std::vector<double>> testOutputs(outputs.size() - trainingSize);

		for (size_t i = 0; i < trainingSize; ++i) {
			trainingInputs[i] = inputs[shuffled_indices[i]];
			trainingOutputs[i] = outputs[shuffled_indices[i]];
		}
		for (size_t i = trainingSize; i < inputs.size(); ++i) {
			testInputs[i - trainingSize] = inputs[shuffled_indices[i]];
			testOutputs[i - trainingSize] = outputs[shuffled_indices[i]];
		}

		// Training and testing
		for (int epoch = 0; epoch < epochs; ++epoch) {
			// Train on training data
			for (size_t i = 0; i < trainingInputs.size(); ++i) {
				nn.forward(trainingInputs[i]);
				nn.backpropagate(trainingOutputs[i]);  // Assuming your NN has a backprop method
			}
		}
	}


}
