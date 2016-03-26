#ifndef _MLLIB_SOFTMAX_REGRESSION_H
#define _MLLIB_SOFTMAX_REGRESSION_H

#include <vector>

using namespace std;

namespace mllib {

class SoftmaxRegression {
public:
    SoftmaxRegression() = default;
    SoftmaxRegression(const size_t epochs, const size_t numClasses,
        const double learningRate, const double epsilon,
        const bool regular, const double lambda);

    ~SoftmaxRegression() {};

    void train(const vector<vector<double>> &data,
        const vector<int> &labels, const size_t batchSize);

    void predict(vector<vector<double>> &data,
        vector<int> &predictLabel, vector<double> &prob);

private:
    vector<vector<double>> weights;
    size_t epochs;
    size_t numClasses;
    double learningRate;
    // 停止条件
    double epsilon;
    // 正则项
    bool regular;
    double lambda;

    SoftmaxRegression(const SoftmaxRegression &reg) = delete;
    const SoftmaxRegression& operator= (const SoftmaxRegression &);

    double logistic(const vector<double> &weight,
        const vector<double> &data);

    double regularize(const vector<vector<double>> &weights);
};

}

#endif