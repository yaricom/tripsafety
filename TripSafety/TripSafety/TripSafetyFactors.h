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

#define USE_ESTIMATORS
//#define USE_REGERESSION

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
//
// -----------------------------------------
//
template <class T>
void concatenate(VC<T> &first, const VC<T> &second) {
    size_t size = second.size();
    if (first.size() < size) {
        // resize
        first.resize(size);
    }
    // update
    for (int i = 0; i < size; i++) {
        first[i] += second[i];
    }
}

template<class bidiiter>
inline bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand()%left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

class RandomSample {
    // generate "m_number" of data samples with the value within the range [0, m_max].
    int m_max;
    int m_number;
    
public:
    RandomSample(int max, int number) : m_max(max), m_number(number) {}
    
    inline VI get_sample_index() {
        // fill vector with indices
        VI re_res(m_max);
        for (int i = 0; i < m_max; ++i)
            re_res[i] = i;
        
        // suffle
        random_unique(re_res.begin(), re_res.end(), m_number);
        
        // resize vector
        re_res.resize(m_number);
        VI(re_res).swap(re_res);
        
        return re_res;
    }
};

struct Node {
    double m_node_value;
    int m_feature_index;
    double m_terminal_left;
    double m_terminal_right;
    
    // Each non-leaf node has a left child and a right child.
    Node *m_left_child = NULL;
    Node *m_right_child = NULL;
    
    // Construction function
    Node(double value, int feature_index, double value_left, double value_right) :
    m_node_value(value), m_feature_index(feature_index), m_terminal_left(value_left), m_terminal_right(value_right) {}
    
private:
    
    Node(Node const &); // non construction-copyable
    Node& operator=(Node const &); // non copyable
};

struct BestSplit {
    // the index of split feature
    int m_feature_index;
    // the calculated node value
    double m_node_value;
    // if false - split failed
    bool m_status;
    
    // construction function
    BestSplit() : m_feature_index(0.0), m_node_value(0.0), m_status(false) {}
};

struct SplitRes {
    VVD m_feature_left;
    VVD m_feature_right;
    double m_left_value = 0.0;
    double m_right_value = 0.0;
    VD m_obs_left;
    VD m_obs_right;
    
    // construction function
    SplitRes() : m_left_value(0.0), m_right_value(0.0) {}
};

struct ListData {
    double m_x;
    double m_y;
    
    ListData(double x, double y) : m_x(x), m_y(y) {}
    
    bool operator < (const ListData& str) const {
        return (m_x < str.m_x);
    }
};

typedef enum _TerminalType {
    AVERAGE, MAXIMAL
}TerminalType;

class RegressionTree {
private:
    // class members
    int m_min_nodes;
    int m_max_depth;
    int m_current_depth;
    TerminalType m_type;
    
public:
    // The root node
    Node *m_root = NULL;
    // The features importance per index
    VI features_importance;
    
    // construction function
    RegressionTree() : m_min_nodes(10), m_max_depth(3), m_current_depth(0), m_type(AVERAGE) {}
    
    // set parameters
    void setMinNodes(int min_nodes) {
        Assert(min_nodes > 3, "The number of terminal nodes is too small: %i", min_nodes);
        m_min_nodes = min_nodes;
    }
    
    void setDepth(int depth) {
        Assert(depth > 0, "Tree depth must be positive: %i", depth);
        m_max_depth = depth;
    }
    
    // get fit value
    double predict(const VD &feature_x) const{
        double re_res = 0.0;
        
        if (!m_root) {
            // failed in building the tree
            return re_res;
        }
        
        Node *current = m_root;
        
        while (true) {
            // current node information
            int c_feature_index = current->m_feature_index;
            double c_node_value = current->m_node_value;
            double c_node_left_value = current->m_terminal_left;
            double c_node_right_value = current->m_terminal_right;
            
            if (feature_x[c_feature_index] < c_node_value) {
                // we should consider left child
                current = current->m_left_child;
                
                if (!current) {
                    re_res = c_node_left_value;
                    break;
                }
            } else {
                // we should consider right child
                current = current->m_right_child;
                
                if (!current) {
                    re_res = c_node_right_value;
                    break;
                }
            }
        }
        
        return re_res;
    }
    
    /*
     *  The method to build regression tree
     */
    void buildRegressionTree(const Matrix &feature_x, const Vector &obs_y) {
        size_t samples_num = feature_x.rows();
        
        Assert(samples_num == obs_y.size() && samples_num != 0,
               "The number of samles does not match with the number of observations or the samples number is 0. Samples: %i", samples_num);
        
        Assert (m_min_nodes * 2 <= samples_num, "The number of samples is too small");
        
        size_t feature_dim = feature_x.cols();
        features_importance.resize(feature_dim, 0);
        
        // build the regression tree
        buildTree(feature_x, obs_y);
    }
    
private:
    
    /**
     *  The following function gets the best split given the data
     */
    BestSplit findOptimalSplit(const Matrix &feature_x, const Vector &obs_y) {
        
        BestSplit split_point;
        
        if (m_current_depth > m_max_depth) {
            return split_point;
        }
        
        size_t samples_num = feature_x.rows();
        
        if (m_min_nodes * 2 > samples_num) {
            // the number of observations in terminals is too small
            return split_point;
        }
        size_t feature_dim = feature_x.cols();
        
        
        double min_err = 0;
        int split_index = -1;
        double node_value = 0.0;
        
        // begin to get the best split information
        for (int loop_i = 0; loop_i < feature_dim; loop_i++){
            // get the optimal split for the loop_index feature
            
            // get data sorted by the loop_i-th feature
            VC<ListData> list_feature;
            for (int loop_j = 0; loop_j < samples_num; loop_j++) {
                list_feature.push_back(ListData(feature_x(loop_j, loop_i), obs_y[loop_j]));
            }
            
            // sort the list
            sort(list_feature.begin(), list_feature.end());
            
            // begin to split
            double sum_left = 0.0;
            double mean_left = 0.0;
            int count_left = 0;
            double sum_right = 0.0;
            double mean_right = 0.0;
            int count_right = 0;
            double current_node_value = 0;
            double current_err = 0.0;
            
            // initialize left
            for (int loop_j = 0; loop_j < m_min_nodes; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                sum_left += fetched_data.m_y;
                count_left++;
            }
            mean_left = sum_left / count_left;
            
            // initialize right
            for (int loop_j = m_min_nodes; loop_j < samples_num; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                sum_right += fetched_data.m_y;
                count_right++;
            }
            mean_right = sum_right / count_right;
            
            // calculate the current error
            // err = ||x_l - mean(x_l)||_2^2 + ||x_r - mean(x_r)||_2^2
            // = ||x||_2^2 - left_count * mean(x_l)^2 - right_count * mean(x_r)^2
            // = constant - left_count * mean(x_l)^2 - right_count * mean(x_r)^2
            // Thus, we only need to check "- left_count * mean(x_l)^2 - right_count * mean(x_r)^2"
            current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
            
            // current node value
            current_node_value = (list_feature[m_min_nodes].m_x + list_feature[m_min_nodes - 1].m_x) / 2;
            
            if (current_err < min_err && current_node_value != list_feature[m_min_nodes - 1].m_x) {
                split_index = loop_i;
                node_value = current_node_value;
                min_err = current_err;
            }
            
            // begin to find the best split point for the feature
            for (int loop_j = m_min_nodes; loop_j <= samples_num - m_min_nodes - 1; loop_j++) {
                ListData fetched_data = list_feature[loop_j];
                double y = fetched_data.m_y;
                sum_left += y;
                count_left++;
                mean_left = sum_left / count_left;
                
                
                sum_right -= y;
                count_right--;
                mean_right = sum_right / count_right;
                
                
                current_err = -1 * count_left * mean_left * mean_left - count_right * mean_right * mean_right;
                // current node value
                current_node_value = (list_feature[loop_j + 1].m_x + fetched_data.m_x) / 2;
                
                if (current_err < min_err && current_node_value != fetched_data.m_x) {
                    split_index = loop_i;
                    node_value = current_node_value;
                    min_err = current_err;
                }
                
            }
        }
        // set the optimal split point
        if (split_index == -1) {
            // failed to split data
            return split_point;
        }
        split_point.m_feature_index = split_index;
        split_point.m_node_value = node_value;
        split_point.m_status = true;
        
        return split_point;
    }
    
    /**
     *  Split data into the left node and the right node based on the best splitting
     *  point.
     */
    SplitRes splitData(const Matrix &feature_x, const Vector &obs_y, const BestSplit &best_split) {
        
        SplitRes split_res;
        
        int feature_index = best_split.m_feature_index;
        double node_value = best_split.m_node_value;
        
        size_t samples_count = obs_y.size();
        for (int loop_i = 0; loop_i < samples_count; loop_i++) {
            VD ith_feature = feature_x[loop_i];
            if (ith_feature[feature_index] < node_value) {
                // append to the left feature
                split_res.m_feature_left.push_back(ith_feature);
                // observation
                split_res.m_obs_left.push_back(obs_y[loop_i]);
            } else {
                // append to the right
                split_res.m_feature_right.push_back(ith_feature);
                split_res.m_obs_right.push_back(obs_y[loop_i]);
            }
        }
        
        // update terminal values
        if (m_type == AVERAGE) {
            double mean_value = 0.0;
            for (double obsL : split_res.m_obs_left) {
                mean_value += obsL;
            }
            mean_value = mean_value / split_res.m_obs_left.size();
            split_res.m_left_value = mean_value;
            
            mean_value = 0.0;
            for (double obsR : split_res.m_obs_right) {
                mean_value += obsR;
            }
            mean_value = mean_value / split_res.m_obs_right.size();
            split_res.m_right_value = mean_value;
            
        } else if (m_type == MAXIMAL) {
            double max_value = 0.0;
            VD::iterator iter = split_res.m_obs_left.begin();
            if (++iter != split_res.m_obs_left.end()) {
                max_value = *iter;
            }
            
            while (++iter != split_res.m_obs_left.end()) {
                double sel_value = *iter;
                if (max_value < sel_value) {
                    max_value = sel_value;
                }
            }
            
            split_res.m_left_value = max_value;
            
            
            // right value
            max_value = 0.0;
            iter = split_res.m_obs_right.begin();
            if (++iter != split_res.m_obs_right.end()) {
                max_value = *iter;
            }
            
            while (++iter != split_res.m_obs_right.end()) {
                double sel_value = *iter;
                if (max_value < sel_value) {
                    max_value = sel_value;
                }
            }
            
            split_res.m_right_value = max_value;
            
        } else {
            // Unknown terminal type
            assert(false);
        }
        
        // return the result
        return split_res;
    }
    
    /*
     *  The following function builds a regression tree from data
     */
    Node* buildTree(const Matrix &feature_x, const Vector &obs_y) {
        
        // obtain the optimal split point
        m_current_depth = m_current_depth + 1;
        
        BestSplit best_split = findOptimalSplit(feature_x, obs_y);
        
        if (!best_split.m_status) {
            if (m_current_depth > 0)
                m_current_depth = m_current_depth - 1;
            
            return NULL;
        }
        
        // update feature importance info
        features_importance[best_split.m_feature_index] += 1;
        
        // split the data
        SplitRes split_data = splitData(feature_x, obs_y, best_split);
        
        // append current value to tree
        Node *new_node = new Node(best_split.m_node_value, best_split.m_feature_index, split_data.m_left_value, split_data.m_right_value);
        
        if (!m_root) {
            m_root = new_node;
            m_current_depth = 0;
            // append left and right side
            m_root->m_left_child = buildTree(split_data.m_feature_left, split_data.m_obs_left); // left
            m_root->m_right_child = buildTree(split_data.m_feature_right, split_data.m_obs_right); // right
        } else {
            // append left and right side
            new_node->m_left_child = buildTree(split_data.m_feature_left, split_data.m_obs_left); // left
            new_node->m_right_child = buildTree(split_data.m_feature_right, split_data.m_obs_right); // right
        }
        if (m_current_depth > 0)
            m_current_depth--;
        
        return new_node;
    }
};

class PredictionForest {
public:
    // class members
    double m_init_value;
    // the tree forest
    VC<RegressionTree> m_trees;
    // the learning rate
    double m_combine_weight;
    
    
    // construction function
    PredictionForest(double learning_rate) : m_init_value(0.0), m_combine_weight(learning_rate) {}
    
    /**
     * The method to make prediction for estimate of function's value from provided features
     *
     * @param feature_x the features to use for prediction
     * @return the estimated function's value
     */
    double predict(const VD &feature_x) {
        double re_res = m_init_value;
        
        if (m_trees.size() == 0) {
            return re_res;
        }
        
        for (int i = 0; i < m_trees.size(); i++) {
            re_res += m_combine_weight * m_trees[i].predict(feature_x);
        }
        
        return re_res;
    }
    
    /**
     * Calculates importance of each feature in input samples
     */
    VI featureImportances() {
        VI importances;
        for (int i = 0; i < m_trees.size(); i++) {
            concatenate(importances, m_trees[i].features_importance);
        }
        return importances;
    }
};


class GradientBoostingMachine {
    // class members
    double m_sampling_size_ratio = 0.5;
    double m_learning_rate = 0.01;
    int m_tree_number = 100;
    
    // tree related parameters
    int m_tree_min_nodes = 10;
    int m_tree_depth = 3;
    
public:
    
    GradientBoostingMachine(double sample_size_ratio, double learning_rate,
                            int tree_number, int tree_min_nodes, int tree_depth) :
    m_sampling_size_ratio(sample_size_ratio), m_learning_rate(learning_rate), m_tree_number(tree_number),
    m_tree_min_nodes(tree_min_nodes), m_tree_depth(tree_depth) {
        // Check the validity of numbers
        Assert(sample_size_ratio > 0 && learning_rate > 0 && tree_number > 0 && tree_min_nodes >= 3 && tree_depth > 0,
               "Wrong parameters");
    }
    
    /**
     * Fits a regression function using the Gradient Boosting Tree method.
     * On success, return function; otherwise, return null.
     *
     * @param input_x the input features
     * @param input_y the ground truth values - one per features row
     */
    PredictionForest *train(const Matrix &input_x, const VD &input_y) {
        
        // initialize forest
        PredictionForest *res_fun = new PredictionForest(m_learning_rate);
        
        // get the samples number
        size_t samples_num = input_y.size();
        
        Assert(samples_num == input_x.rows() && samples_num > 0,
               "Error: The input_x size should not be zero and should match the size of input_y");
        
        // get an initial guess of the function
        double mean_y = 0.0;
        for (double d : input_y) {
            mean_y += d;
        }
        mean_y = mean_y / samples_num;
        res_fun->m_init_value = mean_y;
        
        
        // prepare the iteration
        VD h_value(samples_num);
        // initialize h_value
        int index = 0;
        while (index < samples_num) {
            h_value[index] = mean_y;
            index += 1;
        }
        
        // begin the boosting process
        int iter_index = 0;
        while (iter_index < m_tree_number) {
            
            // calculate the gradient
            VD gradient;
            index = 0;
            for (double d : input_y) {
                gradient.push_back(d - h_value[index]);
                
                // next
                index++;
            }
            
            // begin to sample
            if (m_sampling_size_ratio < 0.99) {
                // sample without replacement
                
                // we need to sample
                RandomSample sampler((int)samples_num, (int) (m_sampling_size_ratio * samples_num));
                
                // get random index
                VI sampled_index = sampler.get_sample_index();
                
                // data for growing trees
                VVD train_x;
                VD train_y;
                
                for (int sel_index : sampled_index) {
                    // assign value
                    train_y.push_back(gradient[sel_index]);
                    train_x.push_back(input_x[sel_index]);
                }
                
                // fit a regression tree
                RegressionTree tree;
                
                if (m_tree_depth > 0) {
                    tree.setDepth(m_tree_depth);
                }
                
                if (m_tree_min_nodes > 0) {
                    tree.setMinNodes(m_tree_min_nodes);
                }
                
                tree.buildRegressionTree(train_x, train_y);
                
                // store tree information
                if (tree.m_root == NULL) {
                    // clear buffer
                    train_x.clear();
                    train_y.clear();
                    continue;
                }
                
                res_fun->m_trees.push_back(tree);
                
                // update h_value information, prepare for the next iteration
                int sel_index = 0;
                while (sel_index < samples_num) {
                    h_value[sel_index] += m_learning_rate * tree.predict(input_x[sel_index]);
                    sel_index++;
                }
                
            } else {
                // use all data
                // fit a regression tree
                RegressionTree tree;
                
                // set parameters if needed
                if (m_tree_depth > 0) {
                    tree.setDepth(m_tree_depth);
                }
                
                if (m_tree_min_nodes > 0) {
                    tree.setMinNodes(m_tree_min_nodes);
                }
                
                tree.buildRegressionTree(input_x, gradient);
                
                if (tree.m_root == NULL) {
                    // cannot update any more
                    break;
                }
                // store tree information
                res_fun->m_trees.push_back(tree);
                
                // update h_value information, prepare for the next iteration
                for (int loop_index = 0; loop_index < samples_num; loop_index++) {
                    h_value[loop_index] += m_learning_rate * tree.predict(input_x[loop_index]);
                }
            }
            
            // next iteration
            iter_index++;
        }
        
        return res_fun;
    }
    
    PredictionForest *learnGradientBoostingRanker(const VC<VD> &input_x, const VC<VD> &input_y, const double tau) {
        PredictionForest *res_fun = new PredictionForest(m_learning_rate);
        
        size_t feature_num = input_x.size();
        
        Assert(feature_num == input_y.size() && feature_num > 0,
               "The size of input_x should be the same as the size of input_y");
        
        VD h_value_x(feature_num, 0);
        VD h_value_y(feature_num, 0);
        
        int iter_index = 0;
        while (iter_index < m_tree_number) {
            
            // in the boosting ranker, randomly select half samples without replacement in each iteration
            RandomSample sampler((int)feature_num, (int) (0.5 * feature_num));
            
            // get random index
            VI sampled_index = sampler.get_sample_index();
            
            VC<VD> gradient_x;
            VD gradient_y;
            
            for (int i = 0; i < sampled_index.size(); i++) {
                int sel_index = sampled_index[i];
                
                gradient_x.push_back(input_x[sel_index]);
                gradient_x.push_back(input_y[sel_index]);
                
                // get sample data
                if (h_value_x[sel_index] < h_value_y[sel_index] + tau) {
                    double neg_gradient = h_value_y[sel_index] + tau - h_value_x[sel_index];
                    gradient_y.push_back(neg_gradient);
                    gradient_y.push_back(-1 * neg_gradient);
                } else {
                    gradient_y.push_back(0.0);
                    gradient_y.push_back(0.0);
                }
            }
            
            // fit a regression tree
            RegressionTree tree;
            //            tree.m_type = MAXIMAL;
            
            tree.buildRegressionTree(gradient_x, gradient_y);
            
            // store tree information
            if (tree.m_root == NULL) {
                continue;
            }
            
            // update information
            res_fun->m_trees.push_back(tree);
            
            double err = 0.0;
            
            for (int loop_index = 0; loop_index < feature_num; loop_index++) {
                h_value_x[loop_index] += m_learning_rate * tree.predict(input_x[loop_index]);
                h_value_y[loop_index] += m_learning_rate * tree.predict(input_y[loop_index]);
                
                if (h_value_x[loop_index] < h_value_y[loop_index] + tau) {
                    err += (h_value_x[loop_index] - h_value_y[loop_index] - tau) *
                    (h_value_x[loop_index] - h_value_y[loop_index] - tau);
                }
            }
            
            iter_index += 1;
        }
        
        
        
        return res_fun;
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
    
public:
    void addValue(double data, double weight) {
        int index = (int)data;
        if (index >= m_Counts.size()) {
            m_Counts.resize(index + 1, 0);
        }
        m_Counts[index] += weight;
        m_SumOfCounts += weight;
    }
    double getProbability(double data) const {
        if (m_SumOfCounts == 0) {
            return 0;
        }
        int index = (int)data;
        if (index > m_Counts.size()) {
            return 0;
        }
//        double p = (double)m_Counts[index] / m_SumOfCounts;
        double p = (double)m_Counts[index];
        return p;
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


double collectFeatures(VD &feats) {
    double val = 0;
    for (int i = 0; i < feats.size(); i++) {
        val += feats[i] * feat_weigths[i]; // 10000;
    }
    return val / feats.size();
}

int featNum = 28;//24;

void createEntryFeatures(const Entry &e, const bool train, VD &feats) {
    feats.push_back(sourceEstimator->getProbability(e.source) * 100); // 0
    feats.push_back(distanceEstimator->getProbability(extractDistanceRange(e)) * 1000); // 1
    feats.push_back(cyclesEstimator->getProbability(e.cycles) * 10); // 2
    feats.push_back(complexityEstimator->getProbability(e.complexity) * 100); // 3
    feats.push_back(cargoEstimator->getProbability(e.cargo)); // 4
    feats.push_back(stopsEstimator->getProbability(e.stops) * 10); // 5
    feats.push_back(startDayEstimator->getProbability(e.start_day) * 10); // 6
    feats.push_back(startMonthEstimator->getProbability(e.start_month)); // 7
    feats.push_back(startDayOfMonthEstimator->getProbability(e.start_day_of_month) * 100); // 8
    feats.push_back(startDayOfWeekEstimator->getProbability(e.start_day_of_week) * 10); // 9
    feats.push_back(startTimeEstimator->getProbability(extractTimeRange(e)) * 100); // 10
    feats.push_back(daysEstimator->getProbability(e.days) * 100); // 11
    feats.push_back(pilotEstimator->getProbability(e.pilot) * 1000); // 12
    feats.push_back(pilot2Estimator->getProbability(e.pilot2) * 1000); // 13
    feats.push_back(pilotExpEstimator->getProbability(e.pilot_exp) * 10); // 14
    feats.push_back(pilotVisitsPrevEstimator->getProbability(e.pilot_visits_prev)); // 15
    feats.push_back(pilotHoursPrevEstimator->getProbability(e.pilot_hours_prev) * 100); // 16
    feats.push_back(pilotDutyHoursPrevEstimator->getProbability(e.pilot_duty_hrs_prev) * 100); // 17
    feats.push_back(pilotDistPrevEstimator->getProbability(extractDistancePrevRange(e))); // 18
    feats.push_back(risk1Estimator->getProbability(extractRouteRisk1(e))); // 19
    feats.push_back(risk2Estimator->getProbability(extractRouteRisk2(e)) * 100); // 20
    feats.push_back(weatherEstimator->getProbability(e.weather) * 10); // 21
    feats.push_back(visibilityEstimator->getProbability(e.visibility) * 100); // 22
    
    feats.push_back(traf0Estimator->getProbability(e.traf0) * 100); // 23
    feats.push_back(traf1Estimator->getProbability(e.traf1) * 100); // 24
    feats.push_back(traf2Estimator->getProbability(e.traf2) * 100); // 25
    feats.push_back(traf3Estimator->getProbability(e.traf3) * 100); // 26
    feats.push_back(traf4Estimator->getProbability(e.traf4) * 100); // 27
    
    if (train) {
        double val = collectFeatures(feats);
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

#ifdef USE_REGERESSION
struct GBTConfig {
    double sampling_size_ratio = 0.5;
    double learning_rate = 0.01;
    int tree_number = 130;
    int tree_min_nodes = 10;
    int tree_depth = 3;
};
#endif

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
#ifdef USE_ESTIMATORS
        rank(trainM, trainEntries, testM, testEntries);
#endif
        
        cerr << "======================" << endl;
        cerr << "======================" << endl;
        sort(testEntries.rbegin(), testEntries.rend(), sortByPrediction);
        for (int i = 0; i < Y; i++) {
            testEntries[i].risk_rank = i + 1;
            Printf("Rank: %i, prediction: %f, events: %i\n", testEntries[i].risk_rank, testEntries[i].predicted, testEntries[i].evt_cnt);
            if (testEntries[i].evt_cnt) {
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
        size_t freqSize = rankFreq.size();
        
        Estimator *estimator = new NormalEstimator(1);
        FORE(i, 0, rankFreq, estimator);

//        WeightSearchEstimator *estimator = new WeightSearchEstimator(1000);
//        estimator->addValues(rankFreq);

        
        // predict
        for (int i = 0; i < Y; i++) {
            double val = collectFeatures(testM[i]);
            
            
//            if (val > 10000) { // 10000 - is the optimum
//                Printf("val: %f ", val);
//                testEntries[i].predicted = 2;
//                printEntry(testEntries[i]);
//                print(testM[i]);
//            } else {
//                testEntries[i].predicted = estimator->getProbability(val);
//            }
            
            if (testEntries[i].dist > 1500 && testEntries[i].visibility < 8 && testEntries[i].pilot_exp < 10 ) {
                val *= 100;
            }
            else if (pilotFreq.getProbability(testEntries[i].pilot) > 5) {
                val *= 50;
            }
//            else if ((testEntries[i].pilot_hours_prev + testEntries[i].pilot_duty_hrs_prev) / (double)testEntries[i].pilot_dist_prev < 0.03) {
//                val *= 20;
//            }
//            else if (testEntries[i].start_time_in_min >= 18 * 60 && testEntries[i].pilot_visits_prev == 0 && (testEntries[i].pilot_hours_prev + testEntries[i].pilot_duty_hrs_prev > 30) && testEntries[i].pilot_dist_prev > 1000) {
//                val *= 100;
//            }
            
            testEntries[i].predicted = val;
            
//            Printf("Id: %i, events: %f, val: %f\n", i, testEntries[i].predicted, val);
            
        }
        
        double finishTime = getTime();
        
        Printf("Rank time: %f\n", finishTime - startTime);
    }
#endif
};

#endif
