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

#ifdef LOCAL
#include "stdc++.h"
#include "estimators.h"
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

static bool LOG_DEBUG = true;
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
    VVD A;
    
    /** Row and column dimensions.
     * m row dimension.
     * n column dimension.
     */
    size_t m, n;
    
    
public:
    
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
    
    Matrix(const VD &vals, const size_t rows) {
        m = rows;
        n = (m != 0 ? vals.size() / m : 0);
        assert (m * n == vals.size());
        
        for (int i = 0; i < m; i++) {
            VD row(n);
            for (int j = 0; j < n; j++) {
                row[j] = vals[i + j * m];
            }
            A.push_back(row);
        }
    }
    
    Matrix(const size_t cols, const VD &vals) {
        n = cols;
        m = (n != 0 ? vals.size() / n : 0);
        assert (m * n == vals.size());
        
        for (int i = 0; i < m; i++) {
            VD row(n);
            for (int j = 0; j < n; j++) {
                row[j] = vals[i * n + j];
            }
            A.push_back(row);
        }
        
    }
    
    size_t rows() const {
        return m;
    }
    
    size_t cols() const {
        return n;
    }
    
    void columnPackedCopy(VD &vals) {
        vals.resize(m * n, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                vals[i + j * m] = A[i][j];
            }
        }
    }
    
    void rowPackedCopy(VD &vals) {
        vals.resize(m * n, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                vals[i * n + j] = A[i][j];
            }
        }
    }
    
    void columnToArray(const int col, VD &vals) {
        assert(col < n);
        vals.resize(m, 0);
        for (int i = 0; i < m; i++) {
            vals[i] = A[i][col];
        }
    }
    
    void addRow(const VD &row) {
        assert(row.size() == n);
        A.push_back(row);
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
    
};

VI sourceFreq;
VI distFreq;
VI cyclesFreq;
VI complexityFreq;
VI cargoFreq;
VI stopsFreq;
VI start_dayFreq;
VI start_monthFreq;
VI start_day_of_monthFreq;
VI start_day_of_weekFreq;
VI start_timeFreq;
VI pilotFreq;
VI pilot2Freq;

VI pilot_expFreq;
VI pilot_visits_prevFreq;

VI pilot_dist_prevFreq;

VI route_risk_1Freq;
VI route_risk_2Freq;
VI weatherFreq;

VI eventsFreq;

inline void collectFrequency(VI &frequencies, const int index, const int value) {
    if (index >= frequencies.size()) {
        frequencies.resize(index + 1, 0);
    }
    frequencies[index] += value;
}

inline int extractTimeRange(const Entry &e) {
    return e.start_time_in_min / 15 + 1;
}

inline int extractDistanceRange(const Entry &e) {
    return e.dist / 50 + 1;
}

inline int extractDistancePrevRange(const Entry &e) {
    return e.pilot_dist_prev / 50 + 1;
}

inline int extractRouteRisk1(const Entry &e) {
    return e.route_risk_1 / 14 + 1;
}

inline int extractRouteRisk2(const Entry &e) {
    return e.route_risk_2 / 20 + 1;
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

Estimator *congestionEstimator = new KernelEstimator(.2);

Estimator *daysEstimator = new KernelEstimator(.01);
Estimator *visibilityEstimator = new KernelEstimator(.1);
Estimator *pilotHoursPrevEstimator = new KernelEstimator(.2);
Estimator *pilotDutyHoursPrevEstimator = new KernelEstimator(.2);

#define FORE(i, a, b, c) for (int i = (a); i < (b).size(); i++) (c)->addValue(i, (b)[i]);

void initFreqEstimators() {
    sourceEstimator = new KernelEstimator(1);
    FORE(i, 0, sourceFreq, sourceEstimator);
    
    distanceEstimator = new KernelEstimator(1);
    FORE(i, 0, distFreq, distanceEstimator);
    
    cyclesEstimator = new DiscreteEstimator((int)cyclesFreq.size(), false);
    FORE(i, 1, cyclesFreq, cyclesEstimator);
    
    complexityEstimator = new DiscreteEstimator((int)complexityFreq.size(), false);
    FORE(i, 0, complexityFreq, complexityEstimator);
    
    cargoEstimator = new DiscreteEstimator((int)cargoFreq.size(), false);
    FORE(i, 1, cargoFreq, cargoEstimator);
    
    stopsEstimator = new DiscreteEstimator((int)stopsFreq.size(), false);//new PoissonEstimator();
    FORE(i, 1, stopsFreq, stopsEstimator);
    
    startDayEstimator = new KernelEstimator(1);
    FORE(i, 1, start_dayFreq, startDayEstimator);
    
    startMonthEstimator = new DiscreteEstimator((int)start_monthFreq.size(), false);//new PoissonEstimator();
    FORE(i, 1, start_monthFreq, startMonthEstimator);
    
    startDayOfMonthEstimator = new KernelEstimator(1);
    FORE(i, 1, start_day_of_monthFreq, startDayOfMonthEstimator);
    
    startDayOfWeekEstimator = new KernelEstimator(1);//new DiscreteEstimator((int)start_day_of_weekFreq.size(), false);
    FORE(i, 1, start_day_of_weekFreq, startDayOfWeekEstimator);
    
    startTimeEstimator = new KernelEstimator(1);
    FORE(i, 0, start_timeFreq, startTimeEstimator);
    
    pilotEstimator = new DiscreteEstimator((int)pilotFreq.size(), false);
    FORE(i, 1, pilotFreq, pilotEstimator);
    
    pilot2Estimator = new DiscreteEstimator((int)pilot2Freq.size(), false);
    FORE(i, 1, pilot2Freq, pilot2Estimator);
    
    pilotExpEstimator = new KernelEstimator(1);
    FORE(i, 0, pilot_expFreq, pilotExpEstimator);
    
    pilotVisitsPrevEstimator = new DiscreteEstimator((int)pilot_visits_prevFreq.size(), false);// new PoissonEstimator();
    FORE(i, 0, pilot_visits_prevFreq, pilotVisitsPrevEstimator);
    
    pilotDistPrevEstimator = new KernelEstimator(1);
    FORE(i, 0, pilot_dist_prevFreq, pilotDistPrevEstimator);
    
    risk1Estimator = new KernelEstimator(1);
    FORE(i, 0, route_risk_1Freq, risk1Estimator);
    
    risk2Estimator = new DiscreteEstimator((int)route_risk_2Freq.size(), false);// new KernelEstimator(1);
    FORE(i, 0, route_risk_2Freq, risk2Estimator);
    
    weatherEstimator = new DiscreteEstimator((int)weatherFreq.size(), false);
    FORE(i, 0, weatherFreq, weatherEstimator);
}

inline double calcCongestionFactor(const Entry &e) {
    double sum = e.traf0 + e.traf1 + e.traf2 + e.traf3 + e.traf4;
    return 100.0 * (e.traf1 + e.traf2 + e.traf3) / sum;
}

void createEntryFeatures(const Entry &e, VD &feats) {
    feats.push_back(sourceEstimator->getProbability(e.source));
    feats.push_back(distanceEstimator->getProbability(extractDistanceRange(e)));
    feats.push_back(cyclesEstimator->getProbability(e.cycles));
    feats.push_back(complexityEstimator->getProbability(e.complexity));
    feats.push_back(cargoEstimator->getProbability(e.cargo));
    feats.push_back(stopsEstimator->getProbability(e.stops));
    feats.push_back(startDayEstimator->getProbability(e.start_day));
    feats.push_back(startMonthEstimator->getProbability(e.start_month));
    feats.push_back(startDayOfMonthEstimator->getProbability(e.start_day_of_month));
    feats.push_back(startDayOfWeekEstimator->getProbability(e.start_day_of_week));
    feats.push_back(startTimeEstimator->getProbability(extractTimeRange(e)));
    
    feats.push_back(daysEstimator->getProbability(e.days));
    
    feats.push_back(pilotEstimator->getProbability(e.pilot));
    feats.push_back(pilot2Estimator->getProbability(e.pilot2));
    
    feats.push_back(pilotExpEstimator->getProbability(e.pilot_exp));
    feats.push_back(pilotVisitsPrevEstimator->getProbability(e.pilot_visits_prev));
    
    feats.push_back(pilotHoursPrevEstimator->getProbability(e.pilot_hours_prev));
    feats.push_back(pilotDutyHoursPrevEstimator->getProbability(e.pilot_duty_hrs_prev));
    feats.push_back(pilotDistPrevEstimator->getProbability(extractDistancePrevRange(e)));
    
    feats.push_back(risk1Estimator->getProbability(extractRouteRisk1(e)));
    feats.push_back(risk2Estimator->getProbability(extractRouteRisk2(e)));
    feats.push_back(weatherEstimator->getProbability(e.weather));
    feats.push_back(visibilityEstimator->getProbability(e.visibility));
    
    feats.push_back(congestionEstimator->getProbability(calcCongestionFactor(e)));
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
    
    // train float estimators
    congestionEstimator->addValue(calcCongestionFactor(e), e.evt_cnt);
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
        
        if (trainData) {
            e.accel_cnt = atoi(vs[index++].c_str());
            e.decel_cnt = atoi(vs[index++].c_str());
            e.speed_cnt = atoi(vs[index++].c_str());
            e.stability_cnt = atoi(vs[index++].c_str());
            e.evt_cnt = atoi(vs[index++].c_str());
            
            collectTrainingData(e);
        }
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

inline void printCollectedFreq() {
    fprintf(stderr, "%s", "Source ID "); printWithIndex(sourceFreq);
    fprintf(stderr, "\n%s", "Distance "); printWithIndex(distFreq);
    fprintf(stderr, "\n%s", "Cycles "); printWithIndex(cyclesFreq);
    fprintf(stderr, "\n%s", "Complexity "); printWithIndex(complexityFreq);
    fprintf(stderr, "\n%s", "Cargo "); printWithIndex(cargoFreq);
    fprintf(stderr, "\n%s", "Stops "); printWithIndex(stopsFreq);
    fprintf(stderr, "\n%s", "Start day "); printWithIndex(start_dayFreq);
    fprintf(stderr, "\n%s", "Start month "); printWithIndex(start_monthFreq);
    fprintf(stderr, "\n%s", "Start day of month "); printWithIndex(start_day_of_monthFreq);
    fprintf(stderr, "\n%s", "Start day of week "); printWithIndex(start_day_of_weekFreq);
    fprintf(stderr, "\n%s", "Start time "); printWithIndex(start_timeFreq);
    fprintf(stderr, "\n%s", "Pilot "); printWithIndex(pilotFreq);
    fprintf(stderr, "\n%s", "Pilot2 "); printWithIndex(pilot2Freq);
    fprintf(stderr, "\n%s", "Pilot experience "); printWithIndex(pilot_expFreq);
    fprintf(stderr, "\n%s", "Pilot visits prev "); printWithIndex(pilot_visits_prevFreq);
    fprintf(stderr, "\n%s", "Pilot dist prev "); printWithIndex(pilot_dist_prevFreq);
    fprintf(stderr, "\n%s", "Route risk1 "); printWithIndex(route_risk_1Freq);
    fprintf(stderr, "\n%s", "Route risk2 "); printWithIndex(route_risk_2Freq);
    fprintf(stderr, "\n%s", "Weather "); printWithIndex(weatherFreq);
    
    fprintf(stderr, "\n%s", "Events "); printWithIndex(eventsFreq);
}

inline void printEstimator(const Estimator &est, const VI &data) {
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
    fprintf(stderr, "\n%s", "Congestion ");
    double maxCongestion = 0;
    for (const Entry &e : data) {
        if (e.evt_cnt > 0) {
            double pred = congestionEstimator->getProbability(calcCongestionFactor(e));
            Printf("%.1f:%f, ", calcCongestionFactor(e), pred);
            if (pred > maxCongestion) {
                maxCongestion = pred;
            }
        }
    }
    Printf("\nMax. days: %f\n", maxCongestion);
    
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

//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//

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
        
        
        
        
        
        VI out;
        
        return out;
    }
};

#endif
