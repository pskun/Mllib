#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include "softmax_regression.h"

using namespace std;
using namespace mllib;

SoftmaxRegression::SoftmaxRegression(const size_t epochs,
        const size_t numClasses,
        const double learningRate,
        const double epsilon,
        const bool regular,
        const double lambda) {
    this->epochs = epochs;
    this->numClasses = numClasses;
    this->learningRate = learningRate;
    this->epsilon = epsilon;
    this->regular = regular;
    this->lambda = lambda;
}

void SoftmaxRegression::train(const vector<vector<double>> &data,
        const vector<int> &labels, const size_t batchSize) {
    size_t numsOfData = data.size();
    size_t numsOfDim = 0;
    if(numsOfData == 0) {
        fprintf(stderr, "nums of data is empty\n");
        return;
    }
    numsOfDim = data[0].size();
    // initialize weights
    weights.reserve(numClasses);
    for(size_t i=0; i<numClasses; i++) {
        vector<double> w(numsOfDim+1, 0.01);
        weights.push_back(w);
    }
    // define some variables
    double prevCost = 0;
    double currCost = 0;
    bool isStop = false;
    size_t numsOfBatches = numsOfData / batchSize;
    if(numsOfData % batchSize) numsOfBatches++;
    vector<double> lg(numClasses);
    vector<vector<double>> gradientSum(numClasses, vector<double>(numsOfDim+1));
    // for each epoch
    for(size_t i=0; i<epochs; ++i) {
        // for each batch
        for(size_t b = 0; b < numsOfBatches; b++) {
            // for each class
            size_t currentBatchSize = min(batchSize, numsOfData-b*batchSize);
            for(size_t c=0; c<numClasses; c++) {
                gradientSum[c].assign(gradientSum.size(), 0);
                // for each sample of training data
                for(size_t j=0; j<currentBatchSize; j++) {
                    size_t dataIndex = b * batchSize + j;
                    double sumLg = 0;
                    // calculate probablity of each class of each sample
                    for(size_t d=0; d<numClasses; d++) {
                        lg[d] = logistic(weights[d], data[dataIndex]);
                        sumLg += lg[d];
                    }
                    // get correct label
                    size_t label = labels[dataIndex];
                    // calculate current cost
                    if(label==c) {
                        currCost += lg[c]>0?log(lg[c] / sumLg):0;
                    }
                    // update weight vector of each class
                    // according to gradient descent
                    lg[c] = (c==label?1:0) - lg[c] / sumLg;
                    gradientSum[c][0] += lg[c];
                    for(size_t k=1; k<=numsOfDim; k++) {
                        gradientSum[c][k] += data[dataIndex][k-1] * lg[c];
                    }
                }
            }
            currCost = -currCost / currentBatchSize;
            if(regular) currCost += regularize(weights) * lambda / 2;
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
            for(size_t c=0; c<numClasses; c++) {
                for(size_t k=0; k<=numsOfDim; k++) {
                    double gradient = -gradientSum[c][k] / currentBatchSize;
                    if(regular) gradient += lambda * weights[c][k];
                    weights[c][k] -= learningRate * gradient;
                }
            }
        }
        if(isStop) break;
        fprintf(stderr, "finish epoch %d.\n", i+1);
    }
}

void SoftmaxRegression::predict(vector<vector<double>> &data,
        vector<int> &predictLabel, vector<double> &prob) {
    size_t numsOfData = data.size();
    // the probability of each class
    vector<double> pr(numClasses, 0);
    for(size_t i=0; i<numsOfData; ++i) {
        for(size_t c=0; c<numClasses; c++) {
            pr[c] = logistic(weights[c], data[i]);
        }
        double maxProb = 0;
        size_t mostProbClass = -1;
        for(size_t c=0; c<numClasses; c++) {
            if(pr[c] > maxProb) {
                maxProb = pr[c];
                mostProbClass = c;
            }
        }
        predictLabel.push_back(mostProbClass);
        prob.push_back(maxProb);
    }
}


double SoftmaxRegression::logistic(const vector<double> &weight,
    const vector<double> &data) {
    double dotProduct = 0;
    size_t numsOfDim = data.size();
    for(size_t i=0; i<numsOfDim; i++) {
        dotProduct += weight[i+1] * data[i];
    }
    dotProduct += weight[0];
    dotProduct = exp(dotProduct);
    return dotProduct;
}

double SoftmaxRegression::regularize(
    const vector<vector<double>> &weights) {
    double ret = 0;
    size_t numsOfDim = weights.size();
    for(size_t c=0; c<numClasses; c++) {
        for(size_t k=0; k<numsOfDim; k++) {
            ret += weights[c][k] * weights[c][k];
        }
    }
    return ret;
}