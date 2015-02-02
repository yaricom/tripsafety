//
//  estimators.h
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 1/31/15.
//  Copyright (c) 2015 Danfoss. All rights reserved.
//

#ifndef TripSafety_estimators_h
#define TripSafety_estimators_h

#include "statistics.h"

/** The small deviation allowed in double comparisons. */
const static double SMALL = 1e-6;

class Estimator {
public:
    /**
     * Get a probability estimate for a value
     *
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    virtual double getProbability(double data, bool inverse) const = 0;
    
    /**
     * Add a new data value to the current estimator.
     *
     * @param data the new data value
     * @param weight the weight assigned to the data value
     */
    virtual void addValue(double data, double weight) = 0;
};

/**
 * Simple probability estimator that places a single Poisson distribution
 * over the observed values.
 */
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
    
    /**
     * Add a new data value to the current estimator.
     *
     * @param data the new data value
     * @param weight the weight assigned to the data value
     */
    void addValue(double data, double weight) {
        m_NumValues += weight;
        m_SumOfValues += data * weight;
        if (m_NumValues != 0) {
            m_Lambda = m_SumOfValues / m_NumValues;
        }
    }
    
    /**
     * Get a probability estimate for a value
     *
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    double getProbability(double data, bool inverse) const {
        double p = Poisson(data);
        if (inverse) {
            p =  1 - p;
        }
        return p;
    }
 
private:
    /**
     * Calculates the log factorial of a number.
     *
     * @param x input number.
     * @return log factorial of x.
     */
    double logFac(double x) const {
        double result = 0;
        for (double i = 2; i <= x; i++) {
            result += std::log(i);
        }
        return result;
    }
    
    /**
     * Returns value for Poisson distribution
     *
     * @param x the argument to the kernel function
     * @return the value for a Poisson kernel
     */
    double Poisson(double x) const {
        return std::exp(-m_Lambda + (x * std::log(m_Lambda)) - logFac(x));
    }
};

/**
 * Simple probability estimator that places a single normal distribution
 * over the observed values.
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 5540 $
 */
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
    /**
     * Constructor that takes a precision argument.
     *
     * @param precision the precision to which numeric values are given. For
     * example, if the precision is stated to be 0.1, the values in the
     * interval (0.25,0.35] are all treated as 0.3.
     */
    NormalEstimator(double precision) {
        m_Precision = precision;
        // Allow at most 3 sd's within one interval
        m_StandardDev = m_Precision / (2 * 3);
    }
    
    /**
     * Add a new data value to the current estimator.
     *
     * @param data the new data value
     * @param weight the weight assigned to the data value
     */
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
    
    /**
     * Get a probability estimate for a value
     *
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    double getProbability(double data, bool inverse) const {
        
        data = round(data);
        double zLower = (data - m_Mean - (m_Precision / 2)) / m_StandardDev;
        double zUpper = (data - m_Mean + (m_Precision / 2)) / m_StandardDev;
        
        double pLower = normalProbability(zLower);
        double pUpper = normalProbability(zUpper);
        
        double p = pUpper - pLower;
        
        if (inverse) {
            p =  1 - p;
        }
        return p;
    }
    
private:
    /**
     * Round a data value using the defined precision for this estimator
     *
     * @param data the value to round
     * @return the rounded data value
     */
    double round(double data) const {
        
        return std::rint(data / m_Precision) * m_Precision;
    }
};

/**
 * Simple kernel density estimator. Uses one gaussian kernel per observed
 * data value.
 */
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
    /**
     * Constructor that takes a precision argument.
     *
     * @param precision the  precision to which numeric values are given. For
     * example, if the precision is stated to be 0.1, the values in the
     * interval (0.25,0.35] are all treated as 0.3.
     */
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
    
    /**
     * Add a new data value to the current estimator.
     *
     * @param data the new data value
     * @param weight the weight assigned to the data value
     */
    void addValue(double data, double weight) {
        
        if (weight == 0) {
            return;
        }
        data = round(data);
        int insertIndex = findNearestValue(data);
        if ((m_NumValues <= insertIndex) || (m_Values[insertIndex] != data)) {
            if (m_NumValues < m_Values.size()) {
                int left = m_NumValues - insertIndex;
                std::copy(m_Values.begin() + insertIndex, m_Values.begin() + insertIndex + left, m_Values.begin() + insertIndex + 1);
                std::copy(m_Weights.begin() + insertIndex, m_Weights.begin() + insertIndex + left, m_Weights.begin() + insertIndex + 1);
                m_Values[insertIndex] = data;
                m_Weights[insertIndex] = weight;
                m_NumValues++;
            } else {
                std::vector<double>newValues(m_Values.size() * 2, 0);
                std::vector<double>newWeights(m_Values.size() * 2, 0);
                int left = m_NumValues - insertIndex;
                std::copy(m_Values.begin(), m_Values.begin() + insertIndex, newValues.begin());
                std::copy(m_Weights.begin(), m_Weights.begin() + insertIndex, newWeights.begin());
                newValues[insertIndex] = data;
                newWeights[insertIndex] = weight;
                std::copy(m_Values.begin() + insertIndex, m_Values.begin() + insertIndex + left, newValues.begin() + insertIndex + 1);
                std::copy(m_Weights.begin() + insertIndex, m_Weights.begin() + insertIndex + left, newWeights.begin() + insertIndex + 1);
                m_NumValues++;
                m_Values = newValues;
                m_Weights = newWeights;
            }
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
    
    /**
     * Get a probability estimate for a value.
     *
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    double getProbability(double data, bool inverse) const {
        
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
        
        if (inverse) {
            p =  1 - p;
        }
        return p;
    }
    
    
private:
    /**
     * Execute a binary search to locate the nearest data value
     *
     * @param the data value to locate
     * @return the index of the nearest data value
     */
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
    
    /**
     * Round a data value using the defined precision for this estimator
     *
     * @param data the value to round
     * @return the rounded data value
     */
    double round(double data) {
        return std::rint(data / m_Precision) * m_Precision;
    }
};

/**
 * Simple symbolic probability estimator based on symbol counts.
 */
class DiscreteEstimator : public Estimator {
    /** Hold the counts */
    std::vector<double> m_Counts;
    /** Hold the sum of counts */
    double m_SumOfCounts;
    
public:
    /**
     * Constructor
     *
     * @param numSymbols the number of possible symbols (remember to include 0)
     * @param laplace if true, counts will be initialised to 1
     */
    DiscreteEstimator(int numSymbols, bool laplace) {
        if (laplace) {
            m_Counts.resize(numSymbols, 1);
            m_SumOfCounts = (double)numSymbols;
        } else {
            m_Counts.resize(numSymbols, 0);
            m_SumOfCounts = 0;
        }
    }
    
    /**
     * Constructor
     *
     * @param nSymbols the number of possible symbols (remember to include 0)
     * @param fPrior value with which counts will be initialised
     */
    DiscreteEstimator(int nSymbols, double fPrior) {
        m_Counts.resize(nSymbols, 0);
        for(int iSymbol = 0; iSymbol < nSymbols; iSymbol++) {
            m_Counts[iSymbol] = fPrior;
        }
        m_SumOfCounts = fPrior * (double) nSymbols;
    }
    
    /**
     * Add a new data value to the current estimator.
     *
     * @param data the new data value
     * @param weight the weight assigned to the data value
     */
    void addValue(double data, double weight) {
        if ((int)data < m_Counts.size()) {
            m_Counts[(int)data] += weight;
            m_SumOfCounts += weight;
        }
    }
    
    /**
     * Get a probability estimate for a value
     *
     * @param data the value to estimate the probability of
     * @return the estimated probability of the supplied value
     */
    double getProbability(double data, bool inverse) const {
        if (m_SumOfCounts == 0) {
            return 0;
        }
        double p = (double)m_Counts[(int)data] / m_SumOfCounts;
        
        if (inverse) {
            p =  1 - p;
        }
        return p;
    }
};

#endif
