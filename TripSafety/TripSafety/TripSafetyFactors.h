//
//  TripSafetyFactors.h
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 1/30/15.
//  Copyright (c) 2015 Danfoss. All rights reserved.
//

#ifndef TripSafety_TripSafetyFactors_h
#define TripSafety_TripSafetyFactors_h


#define LOCAL true

//#define USE_ESTIMATORS
#define USE_REGERESSION

#ifdef LOCAL
#include "stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include <iostream>
#include <sys/time.h>

using namespace std;

#define FOR(i,a,b)  for(int i=(a);i<(b);++i)
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          make_pair
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VF          VC < float >
#define VVF         VC < VF >
#define VS          VC < string >
#define VVS         VC < VS >
#define VE          VC <Entry>

static constexpr double NVL = -1000;
template<class T> void print(VC < T > v) {cerr << "[";if (v.size()) cerr << v[0];FOR(i, 1, v.size()) cerr << ", " << v[i];cerr << "]" << endl;}
template<class T> void printWithIndex(VC < T > v) {cerr << "[";if (v.size()) cerr << "0:" <<  v[0];FOR(i, 1, v.size()) cerr << ", " << i << ":" <<  v[i];cerr << "]" << endl;}

inline VS splt(string s, char c = ',') {
    VS all;
    int p = 0, np;
    while (np = (int)s.find(c, p), np >= 0) {
        if (np != p)
            all.push_back(s.substr(p, np - p));
        else
            all.push_back("");
        p = np + 1;
    }
    if (p < s.size())
        all.push_back(s.substr(p));
    return all;
}

#ifdef LOCAL
static bool LOG_DEBUG = true;
#else
static bool LOG_DEBUG = false;
#endif
/*! the message buffer length */
const int kPrintBuffer = 1 << 12;
inline void Printf(const char *fmt, ...) {
    if (LOG_DEBUG) {
        std::string msg(kPrintBuffer, '\0');
        va_list args;
        va_start(args, fmt);
        vsnprintf(&msg[0], kPrintBuffer, fmt, args);
        va_end(args);
        fprintf(stderr, "%s", msg.c_str());
    }
}

inline void Assert(bool exp, const char *fmt, ...) {
    if (!exp) {
        std::string msg(kPrintBuffer, '\0');
        va_list args;
        va_start(args, fmt);
        vsnprintf(&msg[0], kPrintBuffer, fmt, args);
        va_end(args);
        fprintf(stderr, "AssertError:%s\n", msg.c_str());
        exit(-1);
    }
}

inline double getTime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}
//
// ----------------------------
//
class Vector;

class Matrix {
    
protected:
    size_t m, n;
    
    
public:
    VVD A;
    
    Matrix(const size_t rows, const size_t cols) {
        m = rows;
        n = cols;
        for (int i = 0; i < m; i++) {
            VD row(n, 0);
            A.push_back(row);
        }
    }
    
    Matrix(const VVD &arr) {
        m = arr.size();
        n = arr[0].size();
        for (int i = 0; i < m; i++) {
            assert(arr[i].size() == n);
        }
        A = arr;
    }
    
    size_t rows() const {
        return m;
    }
    
    size_t cols() const {
        return n;
    }
    
    Matrix& subMatrix(int i0, int i1, int j0, int j1) const {
        assert(i0 >= 0 && i0 < i1 && i1 < m && j0 >= 0 && j0 < j1 && j1 < n);
        Matrix *X = new Matrix(i1 - i0 + 1, j1 - j0 + 1);
        for (int i = i0; i <= i1; i++) {
            for (int j = j0; j <= j1; j++) {
                X->A[i - i0][j - j0] = A[i][j];
            }
        }
        return *X;
    }
    
    void columnToArray(const int col, VD &vals) const {
        assert(col < n);
        vals.resize(m, 0);
        for (int i = 0; i < m; i++) {
            vals[i] = A[i][col];
        }
    }
    
    void addRow(const VD &row) {
        assert(row.size() == n);
        A.push_back(row);
        // adjust row counts
        m = A.size();
    }
    
    void addColumn(const VD &col) {
        assert(col.size() == m);
        for (int i = 0; i < m; i++) {
            A[i].push_back(col[i]);
        }
        // inclrease column counts
        n += 1;
    }
    
    Vector& mean();
    
    Vector& variance(const int ddof);
    
    Vector& stdev(const int ddof);
    
    Matrix& transpose() {
        Matrix *X = new Matrix(n, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X->A[j][i] = A[i][j];
            }
        }
        return *X;
    }
    
    Matrix& operator=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = B.A[i][j];
            }
        }
        return *this;
    }
    
    double& operator()(const int i, const int j) {
        assert( i >= 0 && i < m && j >=0 && j < n);
        return A[i][j];
    }
    double operator()(const int i, const int j) const{
        assert( i >= 0 && i < m && j >=0 && j < n);
        return A[i][j];
    }
    VD& operator[](const int row) {
        assert( row >= 0 && row < m);
        return A[row];
    }
    VD operator[](const int row) const{
        assert( row >= 0 && row < m);
        return A[row];
    }
    
    Matrix operator+(const Matrix& B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] + B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator+=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] + B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator-(const Matrix &B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] - B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator-=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] - B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator*(const Matrix &B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] * B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator*=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] * B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator*(const double s) const {
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = s * this->A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator*=(const double s) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = s * this->A[i][j];
            }
        }
        return *this;
    }
    
    Matrix operator/(const Matrix &B) const {
        checkMatrixDimensions(B);
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] / B.A[i][j];
            }
        }
        return X;
    }
    
    Matrix& operator/=(const Matrix &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] = this->A[i][j] / B.A[i][j];
            }
        }
        return *this;
    }
    
    Matrix& operator/=(const double s) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                this->A[i][j] =  this->A[i][j] / s;
            }
        }
        return *this;
    }
    
    Matrix operator/(const double s) const {
        Matrix X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = this->A[i][j] / s;
            }
        }
        return X;
    }
    
    bool operator==(const Matrix &B) const {
        if (m != B.m && n != B.n) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (A[i][j] != B.A[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool similar(const Matrix &B, double diff) {
        if (m != B.m && n != B.n) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (abs(A[i][j] - B.A[i][j]) > diff) {
                    return false;
                }
            }
        }
        return true;
    }
    
    Matrix& matmul(const Matrix &B) const {
        // Matrix inner dimensions must agree.
        assert (B.m == n);
        
        Matrix *X = new Matrix(m, B.n);
        double Bcolj[n];
        for (int j = 0; j < B.n; j++) {
            for (int k = 0; k < n; k++) {
                Bcolj[k] = B.A[k][j];
            }
            for (int i = 0; i < m; i++) {
                VD Arowi = A[i];
                double s = 0;
                for (int k = 0; k < n; k++) {
                    s += Arowi[k] * Bcolj[k];
                }
                X->A[i][j] = s;
            }
        }
        return *X;
    }
    
    
    
protected:
    void checkMatrixDimensions(Matrix B) const {
        assert (B.m != m || B.n != n);
    }
};

class Vector : Matrix {
public:
    
    Vector(int size) : Matrix(size, 1) {}
    
    Vector(VD lst) : Matrix(lst.size(), 1) {
        for (int i = 0; i < lst.size(); i++) {
            A[i][0] = lst[i];
        }
    }
    
    size_t size() const {
        return rows();
    }
    
    double& operator[](const int i) {
        assert( i >= 0 && i < m);
        return A[i][0];
    }
    
    double operator[](const int i) const{
        assert( i >= 0 && i < m);
        return A[i][0];
    }
    
    Vector& operator=(const Vector &B) {
        checkMatrixDimensions(B);
        for (int i = 0; i < m; i++) {
            this->A[i][0] = B.A[i][0];
        }
        return *this;
    }
    
    Vector operator-(const Vector &v) {
        checkMatrixDimensions(v);
        int size = (int)this->size();
        Vector result(size);
        for (int i = 0; i < size; i++) {
            result[i] = result[i] - v.A[i][0];
        }
        return result;
    }
    
    Vector operator+(const Vector &v) {
        checkMatrixDimensions(v);
        int size = (int)this->size();
        Vector result(size);
        for (int i = 0; i < size; i++) {
            result[i] = result[i] + v.A[i][0];
        }
        return result;
    }
    
    Vector& operator/=(const double s) {
        for (int i = 0; i < m; i++) {
            this->A[i][0] = this->A[i][0] / s;
        }
        return *this;
    }
    
    bool operator==(const Vector &B) const {
        if (this->size() != B.size()) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            if (this->A[i][0] != B.A[i][0]) {
                return false;
            }
        }
        return true;
    }
    
    bool similar(const Vector &B, double diff) const {
        if (this->size() != B.size()) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            if (abs(this->A[i][0] - B.A[i][0]) > diff) {
                return false;
            }
        }
        return true;
    }
    
};
Vector& Matrix::mean(){
    size_t cols = this->cols(), rows = this->rows();
    Vector *current = new Vector((int)cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*current)[j] += this->A[i][j];
        }
    }
    (*current) /= rows;
    return *current;
}

Vector& Matrix::variance(const int ddof){
    size_t cols = this->cols(), rows = this->rows();
    assert(ddof < rows);
    
    Vector vmean = this->mean();
    
    Vector *X = new Vector((int)cols);
    double diff;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            diff = this->A[i][j] - vmean[j];
            (*X)[j] += diff * diff;
        }
    }
    
    for (int j = 0; j < cols; j++) {
        (*X)[j] /= (rows - ddof);
    }
    
    return *X;
}

Vector& Matrix::stdev(const int ddof)  {
    Vector &var = this->variance(ddof);
    size_t n = var.size();
    
    for (int j = 0; j < n; j++) {
        var[j] = sqrt(var[j]);
    }
    return var;
}

/**
 * Scales this matrix to have all values centered arround zero with standard deviation = 1
 *
 * @param indices the column's indices for processing
 * @param mat the matrix to be processed
 */
void stdScale(const VI &indices, Matrix &mat) {
    size_t n = mat.cols();
    size_t m = mat.rows();
    
    size_t ind_size = indices.size();
    assert(ind_size <= n);
    
    Vector meanV = mat.mean();
    Vector stdV = mat.stdev(1);
    
    for (int row = 0 ; row < m; row++) {
        for (int col : indices) {
            double X = mat[row][col];
            mat[row][col] = (X - meanV[col]) / stdV[col];
        }
    }
}

//
// -----------------------------------------

#define VERBOSE 0  /* 0 for detailed logs, -1 for summary */

class RidgeRegression {
    // regression coefficients
    VD regrCoef;
    // initial regression coefficients (usually 0)
    VD regrCoefSeed;
    // residual errors per observation
    VD obsErrors;
    
    // the OOB errors calculated if bootstrap used
    VD oobErrors;
    
    // the flag to indicate whether to use bootstrap
    bool useBootstrap;
    
    // the number of observations/samples
    size_t obs = -1;
    // the number of features per observation
    size_t var = -1;
    
    // flag to indicate that data structures initialize
    int init = -1;
    // flag to indicate that approximate regression coeff was calculated
    int seed = -1;
    
    // the calculated MSE after regression complete
    double MSE = 0;
    // the initial MSE before regresion starts
    double MSE_init = 0;
    
public:
    RidgeRegression(bool bootstrap) : useBootstrap(bootstrap) {}
    
    /**
     * Start training regression tree by finding regression coefficients
     * @param train the vector of samples with features per column
     * @param check the vector of dependent variables per sample
     */
    void train(const VVD &train, const VD &check) {
        Assert(train.size() == check.size(), "Samples size should be equal to observations size");
        // find number of variables and observations
        var = train[0].size();
        obs = train.size();
        
        initDataStructures();
        if (useBootstrap) {
            bootstrap(train, check, 20, 0, 100);
        } else {
            initFeaturesSeed(train, check, 2, 5, 4);
            regress(train, check, 0, 100, 1);
        }
    }
    
    /**
     * Do predict based on provided features
     * @param features the features of data sample to find DV
     * @return predicted dependent variable
     */
    double predict(const VD &features) {
        Assert(features.size() == var, "Features size should be equal to train features size, but was: %li", features.size());
        double res = 0;
        for (int i = 0; i < features.size(); i++) {
            res += features[i] * regrCoef[i];
        }
        return res;
    }
    
private:
    
    /**
     * Performs nvalid approximate regression with different starting
     * points to identify an initial approximate solution. This
     * solution will be used as starting point (seed) for the
     * regression.
     *
     * @param mode should be 2 (recommended) or 3; see Regression for details
     * @param nvalid: number of regressions to perform
     * @param niter: number of iterations to use in each regression; should be small here (<10)
     * @param mode: should be 2 (recommended) or 3; see Regression for description
     * Output (global): seed is set to 1 to indicate that Regression must use regrCoefSeed
     * Output: regrCoefSeed: regression coefficients to be used as starting point in Regression
     */
    void initFeaturesSeed(const VVD &train, const VD &check, const long mode, const long nvalid, const long niter) {
        // initialize data structure
        if (seed==1) {
            regrCoefSeed.clear();
        } else {
            seed=1;
        }
        regrCoefSeed.resize(var, 0);
        
        double nsvar, nsvar_max = -1;
        for (int n = 0; n < nvalid; n++) {
            nsvar = regress(train, check, mode, niter, 0);
            if (nsvar > nsvar_max) {
                nsvar_max = nsvar;
                for (int k = 0; k < var; k++) {
                    regrCoefSeed[k] = regrCoef[k];
                }
            }
        }
    }
    
    /**
     * Compute empirical distribution for estimated regression coefficients
     * and reduction in standard deviation of error. Could also be used to
     * compute empirical distribution of error for each observation, to
     * detect outliers.
     *
     * Input (global): var, obs, init (must be set to 1 by using initDataStructures() first)
     * @param niter: number of iterations to use to compute regression coefficients; see Regression.
     * @param nsample: number of samples; Boostrap performs one regression on each sample
     * @param mode: see Regression for details.
     *
     * Output (global): regrCoef: the optimal found regression coefficients
     */
    void bootstrap(const VVD &train, const VD &check, const long nsample, const long mode, const long niter) {
        /* need to run Regress_init first if the data set is new */
        Assert(init == 1, "Must run initDataStructures() fisrt.\n");
        
        VVD bootTrain, bootTest;
        VD bootCheck, bootCheckTest;
        
        // the best found regression coefficients
        VD bestCoef(var, 0);
        
        // array to store selected indices
        VI pick(obs, 0);
        int k, idx, m, oi;
        double oobMSE, oobME, minOOBMSE = numeric_limits<double>::max();
        for (int n = 0; n < nsample; n++) {
            // pick up random observations
            for (k = 0; k < obs; k++) { pick[k] = 0; }
            for (k = 0; k < obs; k++) {
                idx = rand() % obs;
                pick[idx]++;
            }
            
            // create subsample
            for (m = 0; m < obs; m++) {
                if (pick[m] > 0) {
                    // save pick[m] copies of row in data
                    for (k = 0; k < pick[m]; k++) {
                        bootTrain.push_back(train[m]);
                        bootCheck.push_back(check[m]);
                    }
                } else {
                    // save current row as test sample
                    bootTest.push_back(train[m]);
                    bootCheckTest.push_back(check[m]);
                }
            }
            // do regression
            initDataStructures();
            regress(bootTrain, bootCheck, mode, niter, 0);
            
            // find OOB error
            oobMSE = 0;
            for (oi = 0; oi < bootTest.size(); oi++) {
                oobME = predict(bootTest[oi]) - bootCheckTest[oi];
                oobMSE += oobME * oobME;
            }
            oobMSE = sqrt(oobMSE /(double)bootTest.size());
            oobErrors.push_back(oobMSE);
            if (oobMSE < minOOBMSE) {
                oobMSE = minOOBMSE;
                // store current regression coefficients
                bestCoef.swap(regrCoef);
            }
        }
        
        if (LOG_DEBUG) {
            Printf("=================\nOOB errors: ");
            print(oobErrors);
            Printf("\n=================");
        }
        
        // store best regression coefficients as train results
        regrCoef.swap(bestCoef);
    }
    
    /**
     * Performs regression. Must run initDataStructures() first to compute obs,
     * var, initialize init and allocate memory to param. The Regress function
     * returns the reduction in standard deviation of error; this value is
     * between 0 and 1; 0 corresponds to all regression coefficients set to
     * zero; 1 means perfect fit. Regress does not compute confidence
     * intervals for the coefficients (use Boostrap for this purpose).
     *
     * Input (global): var, obs, seed, init
     * @param seed_flag if 1 use initial regression coefficients param_seed
     * to start iterative regression procedure; if seed_flag = 1 then
     * Param_init must be called first to intialize  param_seed and seed.
     *
     * @param niter number of iterations to use to compute regression
     * coefficients; if convergence is slow or erratic then increase
     * niter and perform validation tests with Validation.
     *
     * @param mode determines the type of algorithm used for regression:
     *  mode = 0: visits each variable sequentially starting with first
     *  variable; useful when variables are pre-sorted in such a
     *  way that the first few variables explain most of the
     *  variance.
     *  mode = 1: visits variables in random order; should be the default mode
     *  mode = 2: same as mode = 1, but lambda not randomized
     *
     * @return reduction in standard deviation of error.
     * Output (global): regrCoef: regression coefficients
     */
    double regress(const VVD &features, const VD &dv, const long mode, const long niter, const long seed_flag) {
        /* need to run Regress_init first if the data set is new */
        Assert(init == 1, "Must run initDataStructures() fisrt.\n");
        
        int k, l, i, iter;
        double xd, sp, lambda, val, e_new, resvar = 0;
        
        // calculate initial MSE
        MSE_init = 0;
        for (k = 0; k < obs; k++) {
            val = dv[k];
            MSE_init += val * val;
            obsErrors[k] = val;
        }
        MSE_init = sqrt(MSE_init / obs);
        
        // clear regression coefficients
        for (k = 1; k < var; k++) {
            regrCoef[k] = 0;
        }
        
        // if seed=1 uses initial regressors
        if (seed_flag == 1) {
            Assert(seed == 1, "Must run initFeaturesSeed() first.\n");
            
            for (k = 0; k < var; k++) {
                regrCoef[k] = regrCoefSeed[k];
            }
            
            for (i = 0; i < obs; i++) {
                e_new = 0;
                for (k = 0; k < var; k++) {
                    val = features[i][k];
                    e_new += regrCoef[k] * val;
                }
                // find error
                obsErrors[i] = dv[i] - e_new;
            }
        }
        
        /*
         regression
         */
        for (iter = 0; iter < niter; iter++) {
            if (mode == 0 || mode == 3) {
                // 0: visits each variable sequentially starting with first
                l = 1 + (iter % (var - 1));
            } else {
                // 1: visits variables in random order; should be the default mode
                // 2: same as mode = 1, but lambda not randomized
                l = 1 + rand() % (var - 1);
            }
            
            xd = 0; sp = 0;
            for (k = 0; k < obs; k++) {
                xd += features[k][l] * features[k][l];
                sp += features[k][l] * obsErrors[k];
            }
            Assert(xd != 0, "Empty column found at index: %i", l);
            
            lambda = sp / xd;
            if (mode == 1) {
                lambda = lambda * rand() / (double)RAND_MAX;
            }
            
            // update error
            MSE = 0;
            for (k = 0; k < obs; k++) {
                e_new = obsErrors[k] - lambda * features[k][l];
                MSE += e_new * e_new;
                obsErrors[k] = e_new;
            }
            regrCoef[l] += lambda;
            
            /*
             save results, compute resvar
             */
            
            if (iter % 10 == VERBOSE || iter == niter-1) {
                MSE = sqrt(MSE / obs);
                resvar = 1 - MSE / MSE_init;
                if (LOG_DEBUG) {
                    Printf("REGRESS %d\t%d\t%f\t%f\t", iter, l, lambda, resvar);
                    print(regrCoef);
                }
            }
        }
        return resvar;
    }
    
    /**
     * Reinitialize internal data structures.
     */
    void initDataStructures() {
        // clear data if needed
        if (init == 1) {
            regrCoef.clear();
            obsErrors.clear();
        } else  {
            init = 1;
        }
        
        // resize internal data structures
        regrCoef.resize(var, 0);
        obsErrors.resize(obs, 0);
    }
    
};


//
// -----------------------------------------
//
static const double SQRTH  =  7.07106781186547524401E-1;
static const double MAXLOG =  7.09782712893383996732E2;

double errorFunction(double x);

double p1evl( double x, double coef[], int N ) {
    
    double ans;
    ans = x + coef[0];
    
    for(int i = 1; i < N; i++) ans = ans * x + coef[i];
    
    return ans;
}

double polevl( double x, double coef[], int N ) {
    
    double ans;
    ans = coef[0];
    
    for(int i = 1; i <= N; i++) ans = ans * x + coef[i];
    
    return ans;
}

double errorFunctionComplemented(double a) {
    double x,y,z,p,q;
    
    double P[] = {
        2.46196981473530512524E-10,
        5.64189564831068821977E-1,
        7.46321056442269912687E0,
        4.86371970985681366614E1,
        1.96520832956077098242E2,
        5.26445194995477358631E2,
        9.34528527171957607540E2,
        1.02755188689515710272E3,
        5.57535335369399327526E2
    };
    double Q[] = {
        1.32281951154744992508E1,
        8.67072140885989742329E1,
        3.54937778887819891062E2,
        9.75708501743205489753E2,
        1.82390916687909736289E3,
        2.24633760818710981792E3,
        1.65666309194161350182E3,
        5.57535340817727675546E2
    };
    
    double R[] = {
        5.64189583547755073984E-1,
        1.27536670759978104416E0,
        5.01905042251180477414E0,
        6.16021097993053585195E0,
        7.40974269950448939160E0,
        2.97886665372100240670E0
    };
    double S[] = {
        2.26052863220117276590E0,
        9.39603524938001434673E0,
        1.20489539808096656605E1,
        1.70814450747565897222E1,
        9.60896809063285878198E0,
        3.36907645100081516050E0
    };
    
    if( a < 0.0 )   x = -a;
    else            x = a;
    
    if( x < 1.0 )   return 1.0 - errorFunction(a);
    
    z = -a * a;
    
    if( z < -MAXLOG ) {
        if( a < 0 )  return( 2.0 );
        else         return( 0.0 );
    }
    
    z = std::exp(z);
    
    if( x < 8.0 ) {
        p = polevl( x, P, 8 );
        q = p1evl( x, Q, 8 );
    } else {
        p = polevl( x, R, 5 );
        q = p1evl( x, S, 6 );
    }
    
    y = (z * p)/q;
    
    if( a < 0 ) y = 2.0 - y;
    
    if( y == 0.0 ) {
        if( a < 0 ) return 2.0;
        else        return( 0.0 );
    }
    return y;
}

double errorFunction(double x) {
    double y, z;
    double T[] = {
        9.60497373987051638749E0,
        9.00260197203842689217E1,
        2.23200534594684319226E3,
        7.00332514112805075473E3,
        5.55923013010394962768E4
    };
    double U[] = {
        3.35617141647503099647E1,
        5.21357949780152679795E2,
        4.59432382970980127987E3,
        2.26290000613890934246E4,
        4.92673942608635921086E4
    };
    
    if( std::abs(x) > 1.0 ) return( 1.0 - errorFunctionComplemented(x) );
    z = x * x;
    y = x * polevl( z, T, 4 ) / p1evl( z, U, 5 );
    return y;
}

double normalProbability(double a) {
    double x, y, z;
    
    x = a * SQRTH;
    z = std::abs(x);
    
    if( z < SQRTH ) y = 0.5 + 0.5 * errorFunction(x);
    else {
        y = 0.5 * errorFunctionComplemented(z);
        if( x > 0 )  y = 1.0 - y;
    }
    return y;
}

/** The small deviation allowed in double comparisons. */
const static double SMALL = 1e-6;

class Estimator {
public:
    virtual double getProbability(double data) const = 0;
    
    virtual void addValue(double data, double weight) = 0;
};

class PoissonEstimator : public Estimator {
    /** The number of values seen */
    double m_NumValues;
    /** The sum of the values seen */
    double m_SumOfValues;
    /**
     * The average number of times
     * an event occurs in an interval.
     */
    double m_Lambda;
public:
    void addValue(double data, double weight) {
        m_NumValues += weight;
        m_SumOfValues += data * weight;
        if (m_NumValues != 0) {
            m_Lambda = m_SumOfValues / m_NumValues;
        }
    }
    double getProbability(double data) const {
        double p = Poisson(data);
        return p;
    }
    
private:
    double logFac(double x) const {
        double result = 0;
        for (double i = 2; i <= x; i++) {
            result += std::log(i);
        }
        return result;
    }
    double Poisson(double x) const {
        return std::exp(-m_Lambda + (x * std::log(m_Lambda)) - logFac(x));
    }
};
class NormalEstimator : public Estimator {
    /** The sum of the weights */
    double m_SumOfWeights;
    /** The sum of the values seen */
    double m_SumOfValues;
    /** The sum of the values squared */
    double m_SumOfValuesSq;
    /** The current mean */
    double m_Mean;
    /** The current standard deviation */
    double m_StandardDev;
    /** The precision of numeric values ( = minimum std dev permitted) */
    double m_Precision;
    
public:
    NormalEstimator(double precision) {
        m_Precision = precision;
        // Allow at most 3 sd's within one interval
        m_StandardDev = m_Precision / (2 * 3);
    }
    void addValue(double data, double weight) {
        if (weight == 0) {
            return;
        }
        data = round(data);
        m_SumOfWeights += weight;
        m_SumOfValues += data * weight;
        m_SumOfValuesSq += data * data * weight;
        
        if (m_SumOfWeights > 0) {
            m_Mean = m_SumOfValues / m_SumOfWeights;
            double stdDev = std::sqrt(std::abs(m_SumOfValuesSq - m_Mean * m_SumOfValues) / m_SumOfWeights);
            // If the stdDev ~= 0, we really have no idea of scale yet,
            // so stick with the default. Otherwise...
            if (stdDev > 1e-10) {
                m_StandardDev = std::max(m_Precision / (2 * 3),
                                         // allow at most 3sd's within one interval
                                         stdDev);
            }
        }
    }
    double getProbability(double data) const {
        
        data = round(data);
        double zLower = (data - m_Mean - (m_Precision / 2)) / m_StandardDev;
        double zUpper = (data - m_Mean + (m_Precision / 2)) / m_StandardDev;
        
        double pLower = normalProbability(zLower);
        double pUpper = normalProbability(zUpper);
        
        double p = pUpper - pLower;
        return p;
    }
    
private:
    double round(double data) const {
        
        return std::rint(data / m_Precision) * m_Precision;
    }
};
class KernelEstimator : public Estimator {
    /** Vector containing all of the values seen */
    std::vector<double> m_Values;
    /** Vector containing the associated weights */
    std::vector<double> m_Weights;
    /** Number of values stored in m_Weights and m_Values so far */
    int m_NumValues;
    /** The sum of the weights so far */
    double m_SumOfWeights;
    /** The standard deviation */
    double m_StandardDev;
    /** The precision of data values */
    double m_Precision;
    /** Whether we can optimise the kernel summation */
    bool m_AllWeightsOne;
    /** Maximum percentage error permitted in probability calculations */
    double MAX_ERROR = 0.01;
    
public:
    KernelEstimator(double precision) {
        
        m_Values.resize(50, 0);
        m_Weights.resize(50, 0);
        
        m_NumValues = 0;
        m_SumOfWeights = 0;
        m_AllWeightsOne = true;
        m_Precision = precision;
        // precision cannot be zero
        if (m_Precision < SMALL) m_Precision = SMALL;
        m_StandardDev = m_Precision / (2 * 3);
    }
    void addValue(double data, double weight) {
        
        if (weight == 0) {
            return;
        }
        data = round(data);
        int insertIndex = findNearestValue(data);
        if ((m_NumValues <= insertIndex) || (m_Values[insertIndex] != data)) {
            if (m_NumValues >= m_Values.size()) {
                // resize
                m_Values.resize(m_Values.size() * 2, 0);
                m_Weights.resize(m_Values.size() * 2, 0);
            }
            m_Values.insert(m_Values.begin() + insertIndex, data);
            m_Weights.insert(m_Weights.begin() + insertIndex, data);
            m_NumValues++;
            
            if (weight != 1) {
                m_AllWeightsOne = false;
            }
        } else {
            m_Weights[insertIndex] += weight;
            m_AllWeightsOne = false;
        }
        m_SumOfWeights += weight;
        double range = m_Values[m_NumValues - 1] - m_Values[0];
        if (range > 0) {
            m_StandardDev = std::max(range / std::sqrt(m_SumOfWeights),
                                     // allow at most 3 sds within one interval
                                     m_Precision / (2 * 3));
        }
    }
    double getProbability(double data) const {
        
        double delta = 0, sum = 0, currentProb = 0;
        double zLower = 0, zUpper = 0;
        if (m_NumValues == 0) {
            zLower = (data - (m_Precision / 2)) / m_StandardDev;
            zUpper = (data + (m_Precision / 2)) / m_StandardDev;
            return (normalProbability(zUpper) - normalProbability(zLower));
        }
        double weightSum = 0;
        int start = findNearestValue(data);
        for (int i = start; i < m_NumValues; i++) {
            delta = m_Values[i] - data;
            zLower = (delta - (m_Precision / 2)) / m_StandardDev;
            zUpper = (delta + (m_Precision / 2)) / m_StandardDev;
            currentProb = (normalProbability(zUpper) - normalProbability(zLower));
            sum += currentProb * m_Weights[i];
            weightSum += m_Weights[i];
            if (currentProb * (m_SumOfWeights - weightSum) < sum * MAX_ERROR) {
                break;
            }
        }
        for (int i = start - 1; i >= 0; i--) {
            delta = m_Values[i] - data;
            zLower = (delta - (m_Precision / 2)) / m_StandardDev;
            zUpper = (delta + (m_Precision / 2)) / m_StandardDev;
            currentProb = (normalProbability(zUpper) - normalProbability(zLower));
            sum += currentProb * m_Weights[i];
            weightSum += m_Weights[i];
            if (currentProb * (m_SumOfWeights - weightSum) < sum * MAX_ERROR) {
                break;
            }
        }
        double p = sum / m_SumOfWeights;
        return p;
    }
    
    
private:
    int findNearestValue(double key) const {
        int low = 0;
        int high = m_NumValues;
        int middle = 0;
        while (low < high) {
            middle = (low + high) / 2;
            double current = m_Values[middle];
            if (current == key) {
                return middle;
            }
            if (current > key) {
                high = middle;
            } else if (current < key) {
                low = middle + 1;
            }
        }
        return low;
    }
    double round(double data) {
        return std::rint(data / m_Precision) * m_Precision;
    }
};
class DiscreteEstimator : public Estimator {
    /** Hold the counts */
    VD m_Counts;
    /** Hold the sum of counts */
    double m_SumOfCounts;
    
    NormalEstimator *extrapolator = new NormalEstimator(1);
    
public:
    DiscreteEstimator(int numSymbols, bool laplace) {
        if (laplace) {
            m_Counts.resize(numSymbols, 1);
            m_SumOfCounts = (double)numSymbols;
        } else {
            m_Counts.resize(numSymbols, 0);
            m_SumOfCounts = 0;
        }
    }
    DiscreteEstimator(int nSymbols, double fPrior) {
        m_Counts.resize(nSymbols, 0);
        for(int iSymbol = 0; iSymbol < nSymbols; iSymbol++) {
            m_Counts[iSymbol] = fPrior;
        }
        m_SumOfCounts = fPrior * (double) nSymbols;
    }
    void addValue(double data, double weight) {
        if ((int)data < m_Counts.size()) {
            m_Counts[(int)data] += weight;
            m_SumOfCounts += weight;
        }
        extrapolator->addValue(data, weight);
    }
    double getProbability(double data) const {
        if (m_SumOfCounts == 0) {
            return 0;
        }
        int index = data;
        double p;
        if (index >= m_Counts.size()) {
            p = extrapolator->getProbability(data);
        } else {
            p = (double)m_Counts[index] / m_SumOfCounts;
        }
        
        return p;
    }
};

class MahalanobisEstimator : public Estimator{
    /** The inverse of the covariance matrix */
    Matrix *m_CovarianceInverse;
    /** The determinant of the covariance matrix */
    double m_Determinant;
    /** The difference between the conditioning value and the conditioning mean */
    double m_ConstDelta;
    /** The mean of the values */
    double m_ValueMean;
    
public:
    MahalanobisEstimator(const Matrix &covariance, const double constDelta, const double valueMean) {
        Assert((covariance.rows() == 2) && (covariance.cols() == 2),
               "Wrong covariance matrix dimensions! Rows: %lu, cols: %lu", covariance.rows(), covariance.cols());
        m_CovarianceInverse = NULL;
        
        double a = covariance(0, 0);
        double b = covariance(0, 1);
        double c = covariance(1, 0);
        double d = covariance(1, 1);
        if (a == 0) {
            a = c; c = 0;
            double temp = b;
            b = d; d = temp;
        }
        if (a == 0) {
            return;
        }
        double denom = d - c * b / a;
        if (denom == 0) {
            return;
        }
        
        m_Determinant = covariance(0, 0) * covariance(1, 1) - covariance(1, 0) * covariance(0, 1);
        m_CovarianceInverse = new Matrix(2, 2);
        (*m_CovarianceInverse)(0, 0) = 1.0 / a + b * c / a / a / denom;
        (*m_CovarianceInverse)(0, 1) = -b / a / denom;
        (*m_CovarianceInverse)(1, 0) = -c / a / denom;
        (*m_CovarianceInverse)(1, 1) = 1.0 / denom;
        m_ConstDelta = constDelta;
        m_ValueMean = valueMean;
        
    }
    
    void addValue(double data, double weight) {
        
    }

    double getProbability(double data) const {
        
        double delta = data - m_ValueMean;
        if (m_CovarianceInverse == NULL) {
            return 0;
        }
        return normalKernel(delta);
    }
    
private:
    double normalKernel(double x) const {
        Matrix thisPoint(1, 2);
        thisPoint(0, 0) = x;
        thisPoint(0, 1) = m_ConstDelta;
        
        thisPoint *= (*m_CovarianceInverse);
        thisPoint *= thisPoint.transpose();
        double val = exp(thisPoint(0, 0) / 2) / (sqrt(M_PI * 2) * m_Determinant);
        return val;
    }

};
class NNConditionalEstimator {
    /** Vector containing all of the values seen */
    VD m_Values;
    /** Vector containing all of the conditioning values seen */
    VD m_CondValues;
    /** Vector containing the associated weights */
    VD m_Weights;
    /** The sum of the weights so far */
    double m_SumOfWeights;
    /** Current Conditional mean */
    double m_CondMean;
    /** Current Values mean */
    double m_ValueMean;
    /** Current covariance matrix */
    Matrix *m_Covariance;
    /** Whether we can optimise the kernel summation */
    bool m_AllWeightsOne = true;
    
public:
    void addValue(double data, double given, double weight) {
        
        size_t insertIndex = findNearestPair(given, data);
        
        
        if ((m_Values.size() <= insertIndex) || (m_CondValues[insertIndex] != given) || (m_Values[insertIndex] != data)) {
                m_CondValues.insert(m_CondValues.begin() + insertIndex, given);
                m_Values.insert(m_Values.begin() + insertIndex, data);
                m_Weights.insert(m_Weights.begin() + insertIndex, weight);
                if (weight != 1) {
                    m_AllWeightsOne = false;
                }
            } else {
                double newWeight = m_Weights[insertIndex];
                newWeight += weight;
                m_Weights[insertIndex] = newWeight;
                m_AllWeightsOne = false;      
            }
        m_SumOfWeights += weight;
        
        // Invalidate any previously calculated covariance matrix
        delete m_Covariance;
        m_Covariance = NULL;
    }
    
    double getProbability(double data, double given) {
        if (m_Covariance == NULL) {
            calculateCovariance();
        }
        MahalanobisEstimator estimator(*m_Covariance, given - m_CondMean, m_ValueMean);
        return estimator.getProbability(data);
    }
    
private:
    size_t findNearestPair(double key, double secondaryKey) {
        size_t low = 0;
        size_t high = m_CondValues.size();
        size_t middle = 0;
        while (low < high) {
            middle = (low + high) / 2;
            double current = m_CondValues[middle];
            if (current == key) {
                double secondary = m_Values[middle];
                if (secondary == secondaryKey) {
                    return middle;
                }
                if (secondary > secondaryKey) {
                    high = middle;
                } else if (secondary < secondaryKey) {
                    low = middle + 1;
                }
            }
            if (current > key) {
                high = middle;
            } else if (current < key) {
                low = middle + 1;
            }
        }
        return low;
    }
    
    void calculateCovariance() {
        double sumValues = 0, sumConds = 0;
        for(int i = 0; i < m_Values.size(); i++) {
            sumValues += m_Values[i] * m_Weights[i];
            sumConds += m_CondValues[i] * m_Weights[i];
        }
        m_ValueMean = sumValues / m_SumOfWeights;
        m_CondMean = sumConds / m_SumOfWeights;
        double c00 = 0, c01 = 0, c10 = 0, c11 = 0;
        for(int i = 0; i < m_Values.size(); i++) {
            double x = m_Values[i];
            double y = m_CondValues[i];
            double weight = m_Weights[i];

            c00 += (x - m_ValueMean) * (x - m_ValueMean) * weight;
            c01 += (x - m_ValueMean) * (y - m_CondMean) * weight;
            c11 += (y - m_CondMean) * (y - m_CondMean) * weight;
        }
        c00 /= (m_SumOfWeights - 1.0);
        c01 /= (m_SumOfWeights - 1.0);
        c10 = c01;
        c11 /= (m_SumOfWeights - 1.0);
        m_Covariance = new Matrix(2, 2);
        (*m_Covariance)(0, 0) = c00;
        (*m_Covariance)(0, 1) = c01;
        (*m_Covariance)(1, 0) = c10;
        (*m_Covariance)(1, 1) = c11;
    }
};

class WeightSearchEstimator {
    // the search lenght
    int searchLengthFactor;
    VD funct;
   
public:
    WeightSearchEstimator(const int searchLengthFactor) : searchLengthFactor(searchLengthFactor){}
    
    void addValues(VI &vals) {
        int maxDistance = 0;
        size_t count = vals.size();
        funct.resize(count, 0);
        int prevIndex = 0;
        int distance = 0;
        // find max distance between values
        for (int i = 0; i < count; i++) {
            if (vals[i] > 0) {
                distance = i - prevIndex;
                prevIndex = i;
                if (distance > maxDistance) {
                    maxDistance = distance;
                }
            }
        }
        maxDistance /= searchLengthFactor;
        prevIndex = 0;
        int prevValue = 0, nextValue = 0, nextIndex = 0, distLeft = 0, distRight = 0;
        double leftWeight, rightWeight;
        for (int i = 0; i < count; i++) {
            if (vals[i] > 0) {
                prevIndex = i;
                funct[i] = vals[i];
            } else {
                // find aproximated value
                distLeft = i - prevIndex;
                if (nextIndex <= i) {
                    for (int j = i; j < i + maxDistance; j++) {
                        nextIndex = j;
                        nextValue = vals[j];
                        if (nextValue > 0) {
                            break;
                        }
                    }
                }
                distRight = nextIndex - i;
                leftWeight = (double)distLeft / (distLeft + distRight);
                rightWeight = (double)distRight /  (distLeft + distRight);
                funct[i] = (prevValue * leftWeight + nextValue * rightWeight) / 2.0;
            }
        }
        
        print(funct);
    }
    
    double getProbability(double data) const {
        int index = data;
        if (index >= funct.size() || index < 0) {
            Printf("Value is out of range: %f", data);
            return 0;
        }
        double p = funct[index];
        return p;
    }
};

//
// ----------------------------
//

struct Entry {
    int route_id;
    int source;
    int dist;
    int cycles;
    int complexity;
    int cargo;
    int stops;
    int start_day;
    int start_month;
    int start_day_of_month;
    int start_day_of_week;
    
    string start_time;
    int start_time_in_min;
    
    float days;
    
    // persons
    int pilot;
    int pilot2;
    
    // experience
    int pilot_exp;
    int pilot_visits_prev;
    
    // fatique
    float pilot_hours_prev;
    float pilot_duty_hrs_prev;
    float pilot_dist_prev;
    
    // risk
    int route_risk_1;
    int route_risk_2;
    int weather;
    float visibility;
    
    // traffic
    int traf0;
    int traf1;
    int traf2;
    int traf3;
    int traf4;
    
    // risk event counters
    int accel_cnt;
    int decel_cnt;
    int speed_cnt;
    int stability_cnt;
    int evt_cnt; // total events
    
    // intermediate values
    double predicted;
    int sort_index;
    int risk_rank;
};
void printEntry(const Entry & e) {
    Printf("route_id: %i, source: %i, dist: %i, cycles: %i, complexity: %i, cargo: %i, stops: %i, start_day: %i, start_month: %i, start_day_of_month: %i, start_day_of_week: %i, start_time: %s, days: %.2f, pilot: %i, pilot2: %i, pilot_exp: %i, pilot_visits_prev: %i, pilot_hours_prev: %.2f, pilot_duty_hrs_prev: %.2f, pilot_dist_prev:%.2f, route_risk_1:%i, route_risk_2:%i, weather:%i, visibility:%.1f, traf: %i|%i|%i|%i|%i, evt: %i|%i|%i|%i|%i\n",
           e.route_id, e.source, e.dist, e.cycles, e.complexity, e.cargo, e.stops, e.start_day, e.start_month, e.start_day_of_month, e.start_day_of_week, e.start_time.c_str(), e.days, e.pilot, e.pilot2, e.pilot_exp, e.pilot_visits_prev, e.pilot_hours_prev, e.pilot_duty_hrs_prev, e.pilot_dist_prev, e.route_risk_1, e.route_risk_2, e.weather, e.visibility, e.traf0, e.traf1, e.traf2, e.traf3, e.traf4, e.accel_cnt, e.decel_cnt, e.speed_cnt, e.stability_cnt, e.evt_cnt);
}

class FreqEstimator : public Estimator {
    /** Hold the counts */
    VD m_Counts;
    /** Hold the sum of counts */
    double m_SumOfCounts;
    
    NormalEstimator *extrapolator = new NormalEstimator(1);
    
public:
    void addValue(double data, double weight) {
        int index = (int)data;
        if (index >= m_Counts.size()) {
            m_Counts.resize(index + 1, 0);
        }
        m_Counts[index] += weight;
        m_SumOfCounts += weight;

        extrapolator->addValue(data, weight);
    }
    
    double getProbability(double data) const {
        if (m_SumOfCounts == 0) {
            return 0;
        }
        int index = (int)data;
        if (index > m_Counts.size()) {
            return 0;
        }
//        double p = 1000.0 * (double)m_Counts[index] / m_SumOfCounts;
//        double p = (double)m_Counts[index];
        double p;
        if (index >= m_Counts.size()) {
            p = extrapolator->getProbability(data);
        } else {
            p = (double)m_Counts[index] / m_SumOfCounts;
        }
        return p * 1000.0;
    }
    
    size_t size() const{
        return m_Counts.size();
    }
};

FreqEstimator sourceFreq;
FreqEstimator distFreq;
FreqEstimator cyclesFreq;
FreqEstimator complexityFreq;
FreqEstimator cargoFreq;
FreqEstimator stopsFreq;
FreqEstimator start_dayFreq;
FreqEstimator start_monthFreq;
FreqEstimator start_day_of_monthFreq;
FreqEstimator start_day_of_weekFreq;
FreqEstimator start_timeFreq;
FreqEstimator pilotFreq;
FreqEstimator pilot2Freq;

FreqEstimator pilot_expFreq;
FreqEstimator pilot_visits_prevFreq;

FreqEstimator pilot_dist_prevFreq;

FreqEstimator route_risk_1Freq;
FreqEstimator route_risk_2Freq;
FreqEstimator weatherFreq;

FreqEstimator traf0Freq;
FreqEstimator traf1Freq;
FreqEstimator traf2Freq;
FreqEstimator traf3Freq;
FreqEstimator traf4Freq;

FreqEstimator eventsFreq;

FreqEstimator rankFreq;

inline void collectFrequency(FreqEstimator &est, const int index, const int value) {
    est.addValue(index, value);
}

inline int extractTimeRange(const Entry &e) {
    return e.start_time_in_min / 5 + 1;
}

inline int extractDistanceRange(const Entry &e) {
    return e.dist / 20 + 1;
}

inline int extractDistancePrevRange(const Entry &e) {
    return e.pilot_dist_prev / 20 + 1;
}

inline int extractRouteRisk1(const Entry &e) {
    return e.route_risk_1 / 40 + 1;
}

inline int extractRouteRisk2(const Entry &e) {
    return e.route_risk_2 / 40 + 1;
}

// estimators
Estimator *sourceEstimator;
Estimator *distanceEstimator;
Estimator *cyclesEstimator;
Estimator *complexityEstimator;
Estimator *cargoEstimator;
Estimator *stopsEstimator;
Estimator *startDayEstimator;
Estimator *startMonthEstimator;
Estimator *startDayOfMonthEstimator;
Estimator *startDayOfWeekEstimator;
Estimator *startTimeEstimator;
Estimator *pilotEstimator;
Estimator *pilot2Estimator;
Estimator *pilotExpEstimator;
Estimator *pilotVisitsPrevEstimator;

Estimator *pilotDistPrevEstimator;

Estimator *risk1Estimator;
Estimator *risk2Estimator;
Estimator *weatherEstimator;

Estimator *traf0Estimator;
Estimator *traf1Estimator;
Estimator *traf2Estimator;
Estimator *traf3Estimator;
Estimator *traf4Estimator;

Estimator *daysEstimator = new KernelEstimator(.1);
Estimator *visibilityEstimator = new NormalEstimator(.01);
Estimator *pilotHoursPrevEstimator = new KernelEstimator(.01);
Estimator *pilotDutyHoursPrevEstimator = new KernelEstimator(.01);

Estimator *rankEstimator = new NormalEstimator(.1);

#define FORE(i, a, b, c) for (int i = (a); i < (b).size(); i++) (c)->addValue(i, (b).getProbability(i));

void initFreqEstimators() {
    sourceEstimator = new DiscreteEstimator((int)sourceFreq.size(), false);
    FORE(i, 0, sourceFreq, sourceEstimator);
    
    distanceEstimator = new DiscreteEstimator((int)distFreq.size(), false);
    FORE(i, 0, distFreq, distanceEstimator);
    
    cyclesEstimator = new PoissonEstimator();
    FORE(i, 1, cyclesFreq, cyclesEstimator);
    
    complexityEstimator = new DiscreteEstimator((int)complexityFreq.size(), false);
    FORE(i, 0, complexityFreq, complexityEstimator);
    
    cargoEstimator = new DiscreteEstimator((int)cargoFreq.size(), false);
    FORE(i, 1, cargoFreq, cargoEstimator);
    
    stopsEstimator = new DiscreteEstimator((int)stopsFreq.size(), false);
    FORE(i, 1, stopsFreq, stopsEstimator);
    
    startDayEstimator = new NormalEstimator(1);
    FORE(i, 1, start_dayFreq, startDayEstimator);
    
    startMonthEstimator = new DiscreteEstimator((int)start_monthFreq.size(), false);
    FORE(i, 1, start_monthFreq, startMonthEstimator);
    
    startDayOfMonthEstimator = new KernelEstimator(1);
    FORE(i, 1, start_day_of_monthFreq, startDayOfMonthEstimator);
    
    startDayOfWeekEstimator = new DiscreteEstimator((int)start_day_of_weekFreq.size(), false);
    FORE(i, 1, start_day_of_weekFreq, startDayOfWeekEstimator);
    
    startTimeEstimator = new KernelEstimator(1);
    FORE(i, 0, start_timeFreq, startTimeEstimator);
    
    pilotEstimator = new DiscreteEstimator((int)pilotFreq.size(), false);
    FORE(i, 1, pilotFreq, pilotEstimator);
    
    pilot2Estimator = new DiscreteEstimator((int)pilot2Freq.size(), false);
    FORE(i, 1, pilot2Freq, pilot2Estimator);
    
    pilotExpEstimator = new KernelEstimator(1);
    FORE(i, 0, pilot_expFreq, pilotExpEstimator);
    
    pilotVisitsPrevEstimator = new DiscreteEstimator((int)pilot_visits_prevFreq.size(), false);
    FORE(i, 0, pilot_visits_prevFreq, pilotVisitsPrevEstimator);
    
    pilotDistPrevEstimator = new DiscreteEstimator((int)pilot_dist_prevFreq.size(), false);
    FORE(i, 0, pilot_dist_prevFreq, pilotDistPrevEstimator);
    
    risk1Estimator = new DiscreteEstimator((int)route_risk_1Freq.size(), false);
    FORE(i, 0, route_risk_1Freq, risk1Estimator);
    
    risk2Estimator = new DiscreteEstimator((int)route_risk_2Freq.size(), false);
    FORE(i, 0, route_risk_2Freq, risk2Estimator);
    
    weatherEstimator = new KernelEstimator(1);
    FORE(i, 0, weatherFreq, weatherEstimator);
    
    traf0Estimator = new KernelEstimator(10);
    FORE(i, 0, traf0Freq, traf0Estimator);
    
    traf1Estimator = new KernelEstimator(1);
    FORE(i, 0, traf1Freq, traf1Estimator);
    
    traf2Estimator = new KernelEstimator(1);
    FORE(i, 0, traf2Freq, traf2Estimator);
    
    traf3Estimator = new KernelEstimator(1);
    FORE(i, 0, traf3Freq, traf3Estimator);
    
    traf4Estimator = new KernelEstimator(1);
    FORE(i, 0, traf4Freq, traf4Estimator);
}

inline double calcCongestionFactor(const Entry &e) {
    double sum = e.traf0 + e.traf1 + e.traf2 + e.traf3 + e.traf4;
    if (sum > 0)
        return (e.traf1 + e.traf2 + e.traf3) / sum;
    else
        return 0;
}

double feat_weigths[] = {100, 10, 10, 1, 1, 1, 1, 1, 1, 10, 10, 1, 1000, 1000, 10, 10, 100, 100, 1, 1, 100, 10, 10, 10, 100, 100, 100, 100};


double predictByFeatures(VD &feats) {
    double val = 0;
    for (int i = 0; i < feats.size(); i++) {
        val += feats[i] * feat_weigths[i]; // 10000;
    }
    return val / feats.size();
}

int featNum = 28;//24;

void createEntryFeatures(const Entry &e, const bool train, VD &feats) {
    feats.push_back(sourceEstimator->getProbability(e.source) * 100.0); // 0
    feats.push_back(distanceEstimator->getProbability(extractDistanceRange(e)) * 1000.0); // 1
    feats.push_back(cyclesEstimator->getProbability(e.cycles) * 10.0); // 2
    feats.push_back(complexityEstimator->getProbability(e.complexity) * 100.0); // 3
    feats.push_back(cargoEstimator->getProbability(e.cargo)); // 4
    feats.push_back(stopsEstimator->getProbability(e.stops) * 10.0); // 5
    feats.push_back(startDayEstimator->getProbability(e.start_day) * 10.0); // 6
    feats.push_back(startMonthEstimator->getProbability(e.start_month)); // 7
    feats.push_back(startDayOfMonthEstimator->getProbability(e.start_day_of_month) * 100.0); // 8
    feats.push_back(startDayOfWeekEstimator->getProbability(e.start_day_of_week) * 10.0); // 9
    feats.push_back(startTimeEstimator->getProbability(extractTimeRange(e)) * 100.0); // 10
    feats.push_back(daysEstimator->getProbability(e.days) * 100.0); // 11
    feats.push_back(pilotEstimator->getProbability(e.pilot) * 1000.0); // 12
    feats.push_back(pilot2Estimator->getProbability(e.pilot2) * 1000.0); // 13
    feats.push_back(pilotExpEstimator->getProbability(e.pilot_exp) * 10.0); // 14
    feats.push_back(pilotVisitsPrevEstimator->getProbability(e.pilot_visits_prev)); // 15
    feats.push_back(pilotHoursPrevEstimator->getProbability(e.pilot_hours_prev) * 100.0); // 16
    feats.push_back(pilotDutyHoursPrevEstimator->getProbability(e.pilot_duty_hrs_prev) * 100.0); // 17
    feats.push_back(pilotDistPrevEstimator->getProbability(extractDistancePrevRange(e))); // 18
    feats.push_back(risk1Estimator->getProbability(extractRouteRisk1(e))); // 19
    feats.push_back(risk2Estimator->getProbability(extractRouteRisk2(e)) * 100.0); // 20
    feats.push_back(weatherEstimator->getProbability(e.weather) * 10.0); // 21
    feats.push_back(visibilityEstimator->getProbability(e.visibility) * 100.0); // 22
    
    feats.push_back(traf0Estimator->getProbability(e.traf0) * 100.0); // 23
    feats.push_back(traf1Estimator->getProbability(e.traf1) * 100.0); // 24
    feats.push_back(traf2Estimator->getProbability(e.traf2) * 100.0); // 25
    feats.push_back(traf3Estimator->getProbability(e.traf3) * 100.0); // 26
    feats.push_back(traf4Estimator->getProbability(e.traf4) * 100.0); // 27
    
    if (train) {
        double val = predictByFeatures(feats);
        // to avoid extremes
        collectFrequency(rankFreq, val, e.evt_cnt);
    }
}

void collectTrainingData(const Entry &e) {
    // collect frequencies
    collectFrequency(sourceFreq, e.source, e.evt_cnt);
    collectFrequency(distFreq, extractDistanceRange(e), e.evt_cnt);
    collectFrequency(cyclesFreq, e.cycles, e.evt_cnt);
    collectFrequency(complexityFreq, e.complexity, e.evt_cnt);
    collectFrequency(cargoFreq, e.cargo, e.evt_cnt);
    collectFrequency(stopsFreq, e.stops, e.evt_cnt);
    collectFrequency(start_dayFreq, e.start_day, e.evt_cnt);
    collectFrequency(start_monthFreq, e.start_month, e.evt_cnt);
    collectFrequency(start_day_of_monthFreq, e.start_day_of_month, e.evt_cnt);
    collectFrequency(start_day_of_weekFreq, e.start_day_of_week, e.evt_cnt);
    collectFrequency(start_timeFreq, extractTimeRange(e), e.evt_cnt);
    
    collectFrequency(pilotFreq, e.pilot, e.evt_cnt);
    collectFrequency(pilot2Freq, e.pilot2, e.evt_cnt);
    
    collectFrequency(pilot_expFreq, e.pilot_exp, e.evt_cnt);
    collectFrequency(pilot_visits_prevFreq, e.pilot_visits_prev, e.evt_cnt);
    
    collectFrequency(pilot_dist_prevFreq, extractDistancePrevRange(e), e.evt_cnt);
    
    collectFrequency(route_risk_1Freq, extractRouteRisk1(e), e.evt_cnt);
    collectFrequency(route_risk_2Freq, extractRouteRisk2(e), e.evt_cnt);
    collectFrequency(weatherFreq, e.weather, e.evt_cnt);
    
    collectFrequency(traf0Freq, e.traf0, e.evt_cnt);
    collectFrequency(traf1Freq, e.traf1, e.evt_cnt);
    collectFrequency(traf2Freq, e.traf2, e.evt_cnt);
    collectFrequency(traf3Freq, e.traf3, e.evt_cnt);
    collectFrequency(traf4Freq, e.traf4, e.evt_cnt);
    
    
    // train float estimators
    daysEstimator->addValue(e.days, e.evt_cnt);
    visibilityEstimator->addValue(e.visibility, e.evt_cnt);
    pilotHoursPrevEstimator->addValue(e.pilot_hours_prev, e.evt_cnt);
    pilotDutyHoursPrevEstimator->addValue(e.pilot_duty_hrs_prev, e.evt_cnt);
    
    // store events distribution
    if (e.evt_cnt > 0) {
        collectFrequency(eventsFreq, e.evt_cnt, e.evt_cnt);
    }
}

inline double parseVal(const string &s) {
    return s == "" ? NVL : atof(s.c_str());
}

inline int timeStringToIntMin(const string &s) {
    VS vs = splt(s, ':');
    int hours = atoi(vs[0].c_str());
    int mins = atoi(vs[1].c_str());
    
    return hours * 60 + mins;
}

static void parseInput(const VS &input, bool trainData, VE &output) {
    size_t inputSize = input.size();
    for (int i = 0; i < inputSize; i++) {
        VS vs = splt(input[i], ',');
        int index = 0;
        Entry e;
        e.route_id = atoi(vs[index++].c_str());
        e.source = atoi(vs[index++].c_str());
        e.dist = atoi(vs[index++].c_str());
        e.cycles = atoi(vs[index++].c_str());
        e.complexity = atoi(vs[index++].c_str());
        e.cargo = atoi(vs[index++].c_str());
        e.stops = atoi(vs[index++].c_str());
        e.start_day = atoi(vs[index++].c_str());
        e.start_month = atoi(vs[index++].c_str());
        e.start_day_of_month = atoi(vs[index++].c_str());
        e.start_day_of_week = atoi(vs[index++].c_str());
        
        e.start_time = vs[index++];
        e.start_time_in_min = timeStringToIntMin(e.start_time);
        
        e.days = atof(vs[index++].c_str());
        
        e.pilot = atoi(vs[index++].c_str());
        e.pilot2 = atoi(vs[index++].c_str());
        
        e.pilot_exp = atoi(vs[index++].c_str());
        e.pilot_visits_prev = atoi(vs[index++].c_str());
        
        e.pilot_hours_prev = atoi(vs[index++].c_str());
        e.pilot_duty_hrs_prev = atoi(vs[index++].c_str());
        e.pilot_dist_prev = atoi(vs[index++].c_str());
        
        e.route_risk_1 = atoi(vs[index++].c_str());
        e.route_risk_2 = atoi(vs[index++].c_str());
        e.weather = atoi(vs[index++].c_str());
        e.visibility = atof(vs[index++].c_str());
        
        e.traf0 = atoi(vs[index++].c_str());
        e.traf1 = atoi(vs[index++].c_str());
        e.traf2 = atoi(vs[index++].c_str());
        e.traf3 = atoi(vs[index++].c_str());
        e.traf4 = atoi(vs[index++].c_str());
        
#ifdef LOCAL
        e.accel_cnt = atoi(vs[index++].c_str());
        e.decel_cnt = atoi(vs[index++].c_str());
        e.speed_cnt = atoi(vs[index++].c_str());
        e.stability_cnt = atoi(vs[index++].c_str());
        e.evt_cnt = atoi(vs[index++].c_str());
        if (trainData) {
            collectTrainingData(e);
        }
#else
        if (trainData) {
            e.accel_cnt = atoi(vs[index++].c_str());
            e.decel_cnt = atoi(vs[index++].c_str());
            e.speed_cnt = atoi(vs[index++].c_str());
            e.stability_cnt = atoi(vs[index++].c_str());
            e.evt_cnt = atoi(vs[index++].c_str());
            
            collectTrainingData(e);
        }
#endif
        
        // store entry sort index
        e.sort_index = i;
        
        output[i] = e;
    }
    if (trainData) {
        // initialize estimators
        initFreqEstimators();
    }
}

//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//


inline void printFreqEstimator(const FreqEstimator &est) {
    for (int i = 0; i < est.size(); i++) {
        Printf("%i:%f, ", i, est.getProbability(i));
    }
    cerr << endl;
}

inline void printCollectedFreq() {
    fprintf(stderr, "%s", "Source ID "); printFreqEstimator(sourceFreq);
    fprintf(stderr, "\n%s", "Distance "); printFreqEstimator(distFreq);
    fprintf(stderr, "\n%s", "Cycles "); printFreqEstimator(cyclesFreq);
    fprintf(stderr, "\n%s", "Complexity "); printFreqEstimator(complexityFreq);
    fprintf(stderr, "\n%s", "Cargo "); printFreqEstimator(cargoFreq);
    fprintf(stderr, "\n%s", "Stops "); printFreqEstimator(stopsFreq);
    fprintf(stderr, "\n%s", "Start day "); printFreqEstimator(start_dayFreq);
    fprintf(stderr, "\n%s", "Start month "); printFreqEstimator(start_monthFreq);
    fprintf(stderr, "\n%s", "Start day of month "); printFreqEstimator(start_day_of_monthFreq);
    fprintf(stderr, "\n%s", "Start day of week "); printFreqEstimator(start_day_of_weekFreq);
    fprintf(stderr, "\n%s", "Start time "); printFreqEstimator(start_timeFreq);
    fprintf(stderr, "\n%s", "Pilot "); printFreqEstimator(pilotFreq);
    fprintf(stderr, "\n%s", "Pilot2 "); printFreqEstimator(pilot2Freq);
    fprintf(stderr, "\n%s", "Pilot experience "); printFreqEstimator(pilot_expFreq);
    fprintf(stderr, "\n%s", "Pilot visits prev "); printFreqEstimator(pilot_visits_prevFreq);
    fprintf(stderr, "\n%s", "Pilot dist prev "); printFreqEstimator(pilot_dist_prevFreq);
    fprintf(stderr, "\n%s", "Route risk1 "); printFreqEstimator(route_risk_1Freq);
    fprintf(stderr, "\n%s", "Route risk2 "); printFreqEstimator(route_risk_2Freq);
    fprintf(stderr, "\n%s", "Weather "); printFreqEstimator(weatherFreq);
    
    fprintf(stderr, "\n%s", "Events "); printFreqEstimator(eventsFreq);
}

inline void printEstimator(const Estimator &est, const FreqEstimator &data) {
    for (int i = 0; i < data.size(); i++) {
        Printf("%i:%f, ", i, est.getProbability(i));
    }
    cerr << endl;
}

inline void printTrainedEstimators() {
    fprintf(stderr, "%s", "Source ID "); printEstimator(*sourceEstimator, sourceFreq);
    fprintf(stderr, "\n%s", "Distance "); printEstimator(*distanceEstimator, distFreq);
    fprintf(stderr, "\n%s", "Cycles "); printEstimator(*cyclesEstimator, cyclesFreq);
    fprintf(stderr, "\n%s", "Complexity "); printEstimator(*complexityEstimator, complexityFreq);
    fprintf(stderr, "\n%s", "Cargo "); printEstimator(*cargoEstimator, cargoFreq);
    fprintf(stderr, "\n%s", "Stops "); printEstimator(*stopsEstimator, stopsFreq);
    fprintf(stderr, "\n%s", "Start day "); printEstimator(*startDayEstimator, start_dayFreq);
    fprintf(stderr, "\n%s", "Start month "); printEstimator(*startMonthEstimator, start_monthFreq);
    fprintf(stderr, "\n%s", "Start day of month "); printEstimator(*startDayOfMonthEstimator, start_day_of_monthFreq);
    fprintf(stderr, "\n%s", "Start day of week "); printEstimator(*startDayOfWeekEstimator, start_day_of_weekFreq);
    fprintf(stderr, "\n%s", "Start time "); printEstimator(*startTimeEstimator, start_timeFreq);
    
    fprintf(stderr, "\n%s", "Pilot "); printEstimator(*pilotEstimator, pilotFreq);
    fprintf(stderr, "\n%s", "Pilot2 "); printEstimator(*pilot2Estimator, pilot2Freq);
    
    fprintf(stderr, "\n%s", "Pilot exp "); printEstimator(*pilotExpEstimator, pilot_expFreq);
    fprintf(stderr, "\n%s", "Pilot visits prev "); printEstimator(*pilotVisitsPrevEstimator, pilot_visits_prevFreq);
    
    fprintf(stderr, "\n%s", "Route risk1 "); printEstimator(*risk1Estimator, route_risk_1Freq);
    fprintf(stderr, "\n%s", "Route risk2 "); printEstimator(*risk2Estimator, route_risk_2Freq);
    
    fprintf(stderr, "\n%s", "Weather "); printEstimator(*weatherEstimator, weatherFreq);
    
    fprintf(stderr, "\n%s", "Pilot distance prev "); printEstimator(*pilotDistPrevEstimator, pilot_dist_prevFreq);
}

inline void printTrainedEstimators(const VE &data) {
    
    fprintf(stderr, "\n%s", "Days ");
    double maxDays = 0;
    for (const Entry &e : data) {
        if (e.evt_cnt > 0) {
            double pred = daysEstimator->getProbability(e.days);
            Printf("%.1f:%f, ", e.days, pred);
            if (pred > maxDays) {
                maxDays = pred;
            }
        }
    }
    Printf("\nMax. days: %f\n", maxDays);
    
    fprintf(stderr, "\n%s", "Visibility ");
    double maxVisibility = 0;
    for (const Entry &e : data) {
        if (e.evt_cnt > 0) {
            double pred = visibilityEstimator->getProbability(e.visibility);
            Printf("%.1f:%f, ", e.visibility, pred);
            if (pred > maxVisibility) {
                maxVisibility = pred;
            }
        }
    }
    Printf("\nMax. visibility: %f\n", maxVisibility);
    
    fprintf(stderr, "\n%s", "pilot_hours_prev ");
    double maxHours = 0;
    for (const Entry &e : data) {
        if (e.evt_cnt > 0) {
            double pred = pilotHoursPrevEstimator->getProbability(e.pilot_hours_prev);
            Printf("%.1f:%f, ", e.pilot_hours_prev, pred);
            if (pred > maxHours) {
                maxHours = pred;
            }
        }
    }
    Printf("\nMax. pilot_hours_prev: %f\n", maxHours);
    
    fprintf(stderr, "\n%s", "pilot_duty_hrs_prev ");
    double maxDutyHours = 0;
    for (const Entry &e : data) {
        if (e.evt_cnt > 0) {
            double pred = pilotDutyHoursPrevEstimator->getProbability(e.pilot_duty_hrs_prev);
            Printf("%.1f:%f, ", e.pilot_duty_hrs_prev, pred);
            if (pred > maxDutyHours) {
                maxDutyHours = pred;
            }
        }
    }
    Printf("\nMax. pilot_duty_hrs_prev: %f\n", maxDutyHours);
    
    fprintf(stderr, "\n%s", "pilot_dist_prev ");
    double maxDistPrev = 0;
    for (const Entry &e : data) {
        if (e.evt_cnt > 0) {
            double pred = pilotDutyHoursPrevEstimator->getProbability(e.pilot_dist_prev);
            Printf("%.1f:%f, ", e.pilot_dist_prev, pred);
            if (pred > maxDistPrev) {
                maxDistPrev = pred;
            }
        }
    }
    Printf("\nMax. pilot_dist_prev: %f\n", maxDistPrev);
}

void storeMatrixAsLibSVM(const char* fileName, const Matrix &mat, int classCol = -1) {
    FILE *fp;
    if (!(fp = fopen(fileName, "w"))) {
        throw runtime_error("Failed to open file!");
    }
    
    if (classCol < 0) {
        classCol = (int)mat.cols() - 1;
    }
    assert(classCol < mat.cols());
    // write to the buffer
    for (int row = 0; row < mat.rows(); row++) {
        // write class value first
        double val = mat(row, classCol);
        fprintf(fp, "%f", val);
        int index = 1;
        for (int col = 0; col < mat.cols(); col++) {
            if (col == classCol) {
                // skip
                continue;
            }
            val = mat(row, col);
            if (val) {
                // write only non zero
                fprintf(fp, " %d:%f", index, val);
            }
            index++;
        }
        fprintf(fp, "\n");
    }
    
    // close file
    fclose(fp);
}


//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//

bool sortByPrediction(const Entry &e1, const Entry &e2) {
    return e1.predicted < e2.predicted;
}

bool sortByInitialOrder(const Entry &e1, const Entry &e2) {
    return e1.sort_index < e2.sort_index;
}

class TripSafetyFactors {
    size_t X;
    size_t Y;
    
public:
    VI predict(const VS &trainingData, const VS &testingData) {
        X = trainingData.size();
        Y = testingData.size();
        
        Printf("GB SC Training length: %i, testing length: %i\n", X, Y);
        
        // parse data
        VE trainEntries(X);
        parseInput(trainingData, true, trainEntries);
        VE testEntries(Y);
        parseInput(testingData, false, testEntries);
        
        Assert(trainEntries.size() == X, "Wrong train entries size, expected: %i, found: %lu", X, trainEntries.size());
        Assert(testEntries.size() == Y, "Wrong test entries size, expected: %i, found: %lu", Y, testEntries.size());
        
        if (LOG_DEBUG) {
            printCollectedFreq();
            cerr << "\n===========================================\n" << endl;
            printTrainedEstimators();
            cerr << "\n===========================================\n" << endl;
            printTrainedEstimators(trainEntries);
        }
        
        // collect features
        Matrix trainM(0, featNum + 1);// last is class value
        for (int i = 0; i < trainEntries.size(); i++) {
            VD row;
            createEntryFeatures(trainEntries[i], true, row);
            row.push_back(trainEntries[i].evt_cnt);
            trainM.addRow(row);
        }
        Matrix testM(0, featNum);
        for (const Entry &e : testEntries) {
            VD row;
            createEntryFeatures(e, false, row);
            testM.addRow(row);
        }
        
        
        //        char buff[200];
        //        realpath("./train.libsvm", buff);
        //        Printf("%s", buff);
#ifdef LOCAL
        storeMatrixAsLibSVM("/Users/yaric/train.libsvm", trainM);
#endif
        
        // do classification
        rank(trainM, trainEntries, testM, testEntries);
        
        cerr << "======================" << endl;
        cerr << "======================" << endl;
        sort(testEntries.rbegin(), testEntries.rend(), sortByPrediction);
        for (int i = 0; i < Y; i++) {
            testEntries[i].risk_rank = i + 1;
            Printf("Rank: %i, prediction: %f, events: %i\n", testEntries[i].risk_rank, testEntries[i].predicted, testEntries[i].evt_cnt);
            if (testEntries[i].evt_cnt) {
//                Printf("Rank: %i, prediction: %f, events: %i\n", testEntries[i].risk_rank, testEntries[i].predicted, testEntries[i].evt_cnt);
                printEntry(testEntries[i]);
            }
        }
        
        sort(testEntries.begin(), testEntries.end(), sortByInitialOrder);
        VI result;
        for (int i = 0; i < Y; i++) {
            result.push_back(testEntries[i].risk_rank);
        }
        
        return result;
    }
    
private:
#ifdef USE_ESTIMATORS
    void rank(const Matrix &trainM, const VE &trainEntries,  Matrix &testM, VE &testEntries) {
        cerr << "=========== Rank by Estimators ===========" << endl;
        
        double startTime = getTime();
        
        // predict
        for (int i = 0; i < Y; i++) {
            double val = collectFeatures(testM[i]);
            
            /*// bad and long trip
            if (distFreq.getProbability(extractDistanceRange(testEntries[i])) > 40 && testEntries[i].visibility < 8 && pilot_expFreq.getProbability(testEntries[i].pilot_exp) > 90 ) {
                val *= 100;
            }
            else if (pilotFreq.getProbability(testEntries[i].pilot) > 5) {
                val *= 50;
            }
            // cargo and source
            if (sourceFreq.getProbability(testEntries[i].source) > 60 && cargoFreq.getProbability(testEntries[i].cargo) > 500) {
                val *= 20;
            }*/
            
            // bad and long trip
            if (distFreq.getProbability(extractDistanceRange(testEntries[i])) > 40 && testEntries[i].visibility < 8 && pilot_expFreq.getProbability(testEntries[i].pilot_exp) > 90 ) {
                val *= 100;
            }
            else
                if (pilotFreq.getProbability(testEntries[i].pilot) > 500) {
                val *= 50;
            }
            // cargo and source
//            if (sourceFreq.getProbability(testEntries[i].source) > 0.053 && cargoFreq.getProbability(testEntries[i].cargo) > 0.5) {
//                val *= 20;
//            }

            
            testEntries[i].predicted = val;
            
//            Printf("Id: %i, events: %f, val: %f\n", i, testEntries[i].predicted, val);
            
        }
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
    }
#endif
#ifdef USE_REGERESSION
    
    void rank(const Matrix &trainM, const VE &trainEntries,  Matrix &testM, VE &testEntries) {
        cerr << "=========== Rank by Ridge regression ===========" << endl;
        
        double startTime = getTime();
        
        RidgeRegression ridge(true);
        Matrix trM = trainM.subMatrix(0, (int)trainM.rows() - 1, 0, (int)trainM.cols() - 2);
        stdScale({13}, trM);
        Matrix dvM = trainM.subMatrix(0, (int)trainM.rows() - 1, (int)trainM.cols() - 2, (int)trainM.cols() - 1);
        stdScale({0}, dvM);
        VD dv;
        dvM.columnToArray(0, dv);
        
        ridge.train(trM.A, dv);
        
        // predict
        for (int i = 0; i < Y; i++) {
            testEntries[i].predicted = ridge.predict(testM[i]);
        }
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
    }
#endif
};

#endif
