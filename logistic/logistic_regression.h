#ifndef _MLLIB_LOGISTIC_REGRESSION_H
#define _MLLIB_LOGISTIC_REGRESSION_H

#include <vector>

using namespace std;

namespace mllib {

typedef enum REGULAR {
    NONE = 1,
    L1,
    L2,
} REGULAR;

class LogisticRegression {
public:
    LogisticRegression() = default;
    LogisticRegression(const size_t epochs, const double learningRate,
        const double epsilon, const REGULAR regular, const double lambda);

    ~LogisticRegression() {};

    void train(const vector<vector<double>> &data,
        const vector<int> &labels, const size_t batchSize);

    void predict(vector<vector<double>> &data,
        vector<int> &predictLabel, vector<double> &prob);

private:
    vector<double> weights;
    size_t epochs;
    double learningRate;
    // 停止条件
    double epsilon;
    // 正则项
    REGULAR regular;
    double lambda;

    LogisticRegression(const LogisticRegression &reg) = delete;
    const LogisticRegression& operator= (const LogisticRegression &);

    double logistic(const vector<double> &weight,
        const vector<double> &data);

    double regularize(const vector<double> &weights);
};

}

#endif