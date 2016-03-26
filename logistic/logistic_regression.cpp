#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include "logistic_regression.h"

using namespace std;
using namespace mllib;

LogisticRegression::LogisticRegression(const size_t epochs,
        const double learningRate,
        const double epsilon,
        const REGULAR regular,
        const double lambda) {
    this->epochs = epochs;
    this->learningRate = learningRate;
    this->epsilon = epsilon;
    this->regular = regular;
    this->lambda = lambda;
}

void LogisticRegression::train(const vector<vector<double>> &data,
        const vector<int> &labels, const size_t batchSize) {
    size_t numsOfData = data.size();
    size_t numsOfDim = 0;
    if(numsOfData == 0) {
        fprintf(stderr, "nums of data is empty\n");
        return;
    }
    numsOfDim = data[0].size();
    // initialize weights
    weights.resize(numsOfDim + 1);
    weights.assign(weights.size(), 1);
    // define some variables
    double prevCost = 0;
    double currCost = 0;
    bool isStop = false;
    size_t numsOfBatches = numsOfData / batchSize;
    if(numsOfData % batchSize) numsOfBatches++;
    vector<double> gradient(numsOfDim + 1, 0);
    // for each epoch
    for(size_t i=0; i<epochs; ++i) {
        // for each batch
        for(size_t b = 0; b < numsOfBatches; b++) {
            gradient.assign(gradient.size(), 0);
            size_t currentBatchSize = min(batchSize, numsOfData-b*batchSize);
            // for each sample of training data
            for(size_t j=0; j<currentBatchSize; j++) {
                size_t dataIndex = b * batchSize + j;
                double lg = logistic(weights, data[dataIndex]);
                // get correct label
                size_t label = labels[dataIndex];
                // update weight vector of each class
                // according to gradient descent
                currCost += label==0?-log(1-lg):-log(lg);
                gradient[0] = lg - label;
                if(regular == L2) gradient[0] += lambda / currentBatchSize * weights[0];
                else if(regular == L1) gradient[0] += lambda / currentBatchSize;
                for(size_t k=1; k<=numsOfDim; k++) {
                    gradient[k] += (lg - label) * data[dataIndex][k-1];
                    if(regular == L2) gradient[k] += lambda / currentBatchSize * weights[k];
                    else if(regular == L1) gradient[k] += lambda / currentBatchSize;
                }
            }           
            currCost = currCost / currentBatchSize;
            currCost += regularize(weights) * lambda / 2 / currentBatchSize;
            fprintf(stderr, "finish batch %d of epoch %d, cost %lf\n",
                b+1, i+1, currCost);
            // compare current cost and previous cost
            if(abs(currCost - prevCost) < epsilon) {
                fprintf(stderr,
                    "epoch %d: objective cost vatiation less than epsilon %lf, break.\n", i+1, epsilon);
                isStop = true;
                break;
            }
            prevCost = currCost;
            currCost = 0;
            for(size_t k=0; k<=numsOfDim; k++) {
                double g = gradient[k] / currentBatchSize;
                weights[k] -= learningRate * g;
            }
        }
        if(isStop) break;
        fprintf(stderr, "finish epoch %d.\n", i+1);
    }
}

void LogisticRegression::predict(vector<vector<double>> &data,
        vector<int> &predictLabel, vector<double> &prob) {
    size_t numsOfData = data.size();
    for(size_t i=0; i<numsOfData; ++i) {
        double pr = logistic(weights, data[i]);
        predictLabel.push_back(pr>0.5?1:0);
        prob.push_back(pr);
    }
}


double LogisticRegression::logistic(const vector<double> &weight,
    const vector<double> &data) {
    double dotProduct = 0;
    size_t numsOfDim = data.size();
    for(size_t i=0; i<numsOfDim; i++) {
        dotProduct += weight[i+1] * data[i];
    }
    dotProduct += weight[0];
    dotProduct = 1.0 / (1.0 + exp(-dotProduct));
    return dotProduct;
}

double LogisticRegression::regularize(const vector<double> &weights) {
    double ret = 0;
    if(regular == NONE) return ret;
    size_t numsOfDim = weights.size();
    for(size_t k=0; k<numsOfDim; k++) {
        if(regular == L2) ret += weights[k] * weights[k];
        else if(regular == L1) ret += weights[k];
    }
    return ret;
}