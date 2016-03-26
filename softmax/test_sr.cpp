#include "softmax_regression.h"
#include <iostream>

using namespace std;
using namespace mllib;

int main() {
    vector<vector<double>> data = {
        {46,77,23},
        {48,74,22},
        {34,76,21},
        {35,75,24},
        {34,77,25},
        {55,76,21},
        {56,74,22},
        {55,72,22},
    };

    vector<int> labels = {
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
    };

    const size_t epochs = 1000;
    const size_t numClasses = 3;
    const double learningRate = 0.1;
    const double epsilon = 0.00001;
    const bool regular = false;
    const double lamdba = 0.1;
    SoftmaxRegression* sr = new SoftmaxRegression(
        epochs, numClasses, learningRate, epsilon, regular, lamdba);

    const size_t batchSize = 100;
    sr->train(data, labels, batchSize);

    vector<int> predictLabel;
    vector<double> prob;
    sr->predict(data, predictLabel, prob);
    delete sr;

    size_t labelSize = labels.size();
    for(size_t i=0; i<labelSize; i++) {
        cout << labels[i] << "\t" << predictLabel[i] << endl;
    }
}