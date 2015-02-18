//
//  TripSafetyFactorsPsycho.h
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 2/17/15.
//  Copyright (c) 2015 Danfoss. All rights reserved.
//

#ifndef TripSafety_TripSafetyFactorsPsycho_h
#define TripSafety_TripSafetyFactorsPsycho_h

#define LOCAL true

#ifdef LOCAL
#include "stdc++.h"
#else
#include <bits/stdc++.h>
#endif

#include <sys/time.h>
//#include <emmintrin.h>

bool modeAll = true;//false;
const double SPLIT_SIZE = 0.66;
const double MAX_EVENT = 2;
// const double MAX_EVENT = 2;

#ifdef LOCAL
int THREADS_NO = 4;
#else
int THREADS_NO = 1;
#endif

using namespace std;

#define INLINE   inline __attribute__ ((always_inline))
#define NOINLINE __attribute__ ((noinline))

#define ALIGNED __attribute__ ((aligned(16)))

#define likely(x)   __builtin_expect(!!(x),1)
#define unlikely(x) __builtin_expect(!!(x),0)

#define SSELOAD(a)     _mm_load_si128((__m128i*)&a)
#define SSESTORE(a, b) _mm_store_si128((__m128i*)&a, b)

#define FOR(i,a,b)  for(int i=(a);i<(b);++i)
#define REP(i,a)    FOR(i,0,a)
#define ZERO(m)     memset(m,0,sizeof(m))
#define ALL(x)      x.begin(),x.end()
#define PB          push_back
#define S           size()
#define LL          long long
#define ULL         unsigned long long
#define LD          long double
#define MP          make_pair
#define X           first
#define Y           second
#define VC          vector
#define PII         pair <int, int>
#define VI          VC < int >
#define VVI         VC < VI >
#define VVVI        VC < VVI >
#define VPII        VC < PII >
#define VD          VC < double >
#define VVD         VC < VD >
#define VVVD        VC < VVD >
#define VS          VC < string >
#define VVS         VC < VS >
#define DB(a)       cerr << #a << ": " << (a) << endl;

template<class T> void print(VC < T > v) {cerr << "[";if (v.S) cerr << v[0];FOR(i, 1, v.S) cerr << ", " << v[i];cerr << "]" << endl;}
template<class T> string i2s(T x) {ostringstream o; o << x; return o.str();}
VS splt(string s, char c = ' ') {VS all; int p = 0, np; while (np = s.find(c, p), np >= 0) {if (np != p) all.PB(s.substr(p, np - p)); p = np + 1;} if (p < s.S) all.PB(s.substr(p)); return all;}

double getTime() {
    timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


struct RNG {
    unsigned int MT[624];
    int index;
    
    RNG(int seed = 1) {
        init(seed);
    }
    
    void init(int seed = 1) {
        MT[0] = seed;
        FOR(i, 1, 624) MT[i] = (1812433253UL * (MT[i-1] ^ (MT[i-1] >> 30)) + i);
        index = 0;
    }
    
    void generate() {
        const unsigned int MULT[] = {0, 2567483615UL};
        REP(i, 227) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i+397] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        FOR(i, 227, 623) {
            unsigned int y = (MT[i] & 0x8000000UL) + (MT[i+1] & 0x7FFFFFFFUL);
            MT[i] = MT[i-227] ^ (y >> 1);
            MT[i] ^= MULT[y&1];
        }
        unsigned int y = (MT[623] & 0x8000000UL) + (MT[0] & 0x7FFFFFFFUL);
        MT[623] = MT[623-227] ^ (y >> 1);
        MT[623] ^= MULT[y&1];
    }
    
    unsigned int rand() {
        if (index == 0) {
            generate();
        }
        
        unsigned int y = MT[index];
        y ^= y >> 11;
        y ^= y << 7  & 2636928640UL;
        y ^= y << 15 & 4022730752UL;
        y ^= y >> 18;
        index = index == 623 ? 0 : index + 1;
        return y;
    }
    
    INLINE int next() {
        return rand();
    }
    
    INLINE int next(int x) {
        return rand() % x;
    }
    
    INLINE int next(int a, int b) {
        return a + (rand() % (b - a));
    }
    
    INLINE double nextDouble() {
        return (rand() + 0.5) * (1.0 / 4294967296.0);
    }
};

struct TreeNode {
    int level;
    int feature;
    double value;
    double result;
    int left;
    int right;
    
    TreeNode() {
        level = -1;
        feature = -1;
        value = 0;
        result = 0;
        left = -1;
        right = -1;
    }
};

struct MLConfig {
    static const int MSE = 0;
    static const int MCE = 1;
    static const int MAE = 2;
    static const int CUSTOM = 3;
    
    VI randomFeatures = {5};
    VI randomPositions = {2};
    int featuresIgnored = 0;
    int groupFeature = -1;
    VI groups = {};
    int maxLevel = 100;
    int maxNodeSize = 1;
    int maxNodes = 0;
    int threadsNo = 1;
    int treesNo = 1;
    double bagSize = 1.0;
    double timeLimit = 0;
    int lossFunction = MSE;
    bool useBootstrapping = true;
    bool computeImportances = false;
    bool saveChosenSamples = false;
    
    //Boosting
    bool useLineSearch = false;
    double shrinkage = 0.1;
};

class DecisionTree {
public:
    VC<TreeNode> nodes;
    VD importances;
    VI samplesChosen;
    
private:
    template <class T> INLINE T customLoss(T x) {
        // const double ALPHA = .25;
        // return abs(x) < ALPHA ? x * x / 2.0 : ALPHA * (abs(x) - ALPHA / 2);
        return abs(x) * sqrt(abs(x));
    }
    
    
public:
    
    DecisionTree() { }
    
    template <class T> INLINE DecisionTree(VC<VC<T>> &features, VC<T> &results, MLConfig &config, int seed, const int USE_WEIGHTS, const int SCORING_FUNCTION) {
        RNG r(seed);
        
        if (config.computeImportances) {
            importances = VD(features[0].S);
        }
        
        VI chosenGroups(features.S);
        if (config.groupFeature == -1 && config.groups.S == 0) {
            REP(i, (int)(features.S * config.bagSize)) chosenGroups[r.next(features.S)]++;
        } else if (config.groupFeature != -1) {
            assert(config.groupFeature >= 0 && config.groupFeature < features.S);
            unordered_map<T, int> groups;
            int groupsNo = 0;
            REP(i, features.S) if (!groups.count(features[i][config.groupFeature])) {
                groups[features[i][config.groupFeature]] = groupsNo++;
            }
            VI groupSize(groupsNo);
            REP(i, (int)(groupsNo * config.bagSize)) groupSize[r.next(groupsNo)]++;
            REP(i, features.S) chosenGroups[i] = groupSize[groups[features[i][config.groupFeature]]];
        } else {
            assert(config.groups.S == features.S);
            int groupsNo = 0;
            for (int x : config.groups) groupsNo = max(groupsNo, x + 1);
            VI groupSize(groupsNo);
            REP(i, (int)(groupsNo * config.bagSize)) groupSize[r.next(groupsNo)]++;
            REP(i, features.S) chosenGroups[i] = groupSize[config.groups[i]];
        }
        
        int bagSize = 0;
        REP(i, features.S) if (chosenGroups[i]) bagSize++;
        
        VI bag(bagSize);
        VI weight(features.S);
        
        int pos = 0;
        
        REP(i, features.S) {
            weight[i] = config.useBootstrapping ? chosenGroups[i] : min(1, chosenGroups[i]);
            if (chosenGroups[i]) bag[pos++] = i;
        }
        
        if (config.saveChosenSamples) samplesChosen = chosenGroups;
        
        TreeNode root;
        root.level = 0;
        root.left = 0;
        root.right = pos;
        nodes.PB(root);
        
        queue<int> stack;
        stack.push(0);
        
        while (!stack.empty()) {
            bool equal = true;
            
            int curNode = stack.front(); stack.pop();
            
            int bLeft = nodes[curNode].left;
            int bRight = nodes[curNode].right;
            int bSize = bRight - bLeft;
            
            int totalWeight = 0;
            T totalSum = 0;
            T total2Sum = 0;
            FOR(i, bLeft, bRight) {
                if (USE_WEIGHTS) {
                    totalSum += results[bag[i]] * weight[bag[i]];
                    totalWeight += weight[bag[i]];
                    if (SCORING_FUNCTION == MLConfig::MSE) total2Sum += results[bag[i]] * results[bag[i]] * weight[bag[i]];
                } else {
                    totalSum += results[bag[i]];
                    if (SCORING_FUNCTION == MLConfig::MSE) total2Sum += results[bag[i]] * results[bag[i]];
                }
            }
            assert(bSize > 0);
            
            if (!USE_WEIGHTS) totalWeight = bSize;
            
            FOR(i, bLeft + 1, bRight) if (results[bag[i]] != results[bag[i - 1]]) {
                equal = false;
                break;
            }
            
            if (equal || bSize <= config.maxNodeSize || nodes[curNode].level >= config.maxLevel || config.maxNodes && nodes.S >= config.maxNodes) {
                nodes[curNode].result = totalSum / totalWeight;
                continue;
            }
            
            int bestFeature = -1;
            int bestLeft = 0;
            int bestRight = 0;
            T bestValue = 0;
            T bestLoss = 1e99;
            
            const int randomFeatures = config.randomFeatures[min(nodes[curNode].level, (int)config.randomFeatures.S - 1)];
            REP(i, randomFeatures) {
                
                int featureID = config.featuresIgnored + r.next(features[0].S - config.featuresIgnored);
                
                T vlo, vhi;
                vlo = vhi = features[bag[bLeft]][featureID];
                FOR(j, bLeft + 1, bRight) {
                    vlo = min(vlo, features[bag[j]][featureID]);
                    vhi = max(vhi, features[bag[j]][featureID]);
                }
                if (vlo == vhi) continue;
                
                const int randomPositions = config.randomPositions[min(nodes[curNode].level, (int)config.randomPositions.S - 1)];
                REP(j, randomPositions) {
                    T splitValue = features[bag[bLeft + r.next(bSize)]][featureID];
                    if (splitValue == vlo) {
                        j--;
                        continue;
                    }
                    
                    if (SCORING_FUNCTION == MLConfig::MSE) {
                        T sumLeft = 0;
                        T sum2Left = 0;
                        int totalLeft = 0;
                        FOR(k, bLeft, bRight) {
                            int p = bag[k];
                            if (features[p][featureID] < splitValue) {
                                if (USE_WEIGHTS) {
                                    sumLeft += results[p] * weight[p];
                                    sum2Left += results[p] * results[p] * weight[p];
                                } else {
                                    sumLeft += results[p];
                                    sum2Left += results[p] * results[p];
                                }
                                totalLeft++;
                            }
                        }
                        
                        T sumRight = totalSum - sumLeft;
                        T sum2Right = total2Sum - sum2Left;
                        int totalRight = bSize - totalLeft;
                        
                        if (totalLeft == 0 || totalRight == 0)
                            continue;
                        
                        T loss = sum2Left - sumLeft * sumLeft / totalLeft + sum2Right - sumRight * sumRight / totalRight;
                        
                        if (loss < bestLoss) {
                            bestLoss = loss;
                            bestValue = splitValue;
                            bestFeature = featureID;
                            bestLeft = totalLeft;
                            bestRight = totalRight;
                            if (loss == 0) goto outer;
                        }
                    } else {
                        T sumLeft = 0;
                        int totalLeft = 0;
                        int weightLeft = 0;
                        FOR(k, bLeft, bRight) {
                            int p = bag[k];
                            if (features[p][featureID] < splitValue) {
                                if (USE_WEIGHTS) {
                                    sumLeft += results[p] * weight[p];
                                    weightLeft += weight[p];
                                } else {
                                    sumLeft += results[p];
                                }
                                totalLeft++;
                            }
                        }
                        
                        if (!USE_WEIGHTS) weightLeft = totalLeft;
                        
                        T sumRight = totalSum - sumLeft;
                        int weightRight = totalWeight - weightLeft;
                        int totalRight = bSize - totalLeft;
                        
                        if (totalLeft == 0 || totalRight == 0)
                            continue;
                        
                        T meanLeft = sumLeft / weightLeft;
                        T meanRight = sumRight / weightRight;
                        T loss = 0;
                        
                        if (SCORING_FUNCTION == MLConfig::MCE) {
                            FOR(k, bLeft, bRight) {
                                int p = bag[k];
                                if (features[p][featureID] < splitValue) {
                                    loss += abs(results[p] - meanLeft)  * (results[p] - meanLeft)  * (results[p] - meanLeft)  * weight[p];
                                } else {
                                    loss += abs(results[p] - meanRight) * (results[p] - meanRight) * (results[p] - meanRight) * weight[p];
                                }
                                if (loss > bestLoss) break; //OPTIONAL
                            }
                        } else if (SCORING_FUNCTION == MLConfig::MAE) {
                            FOR(k, bLeft, bRight) {
                                int p = bag[k];
                                if (features[p][featureID] < splitValue) {
                                    loss += abs(results[p] - meanLeft)  * weight[p];
                                } else {
                                    loss += abs(results[p] - meanRight) * weight[p];
                                }
                                if (loss > bestLoss) break; //OPTIONAL
                            }
                        } else if (SCORING_FUNCTION == MLConfig::CUSTOM) {
                            FOR(k, bLeft, bRight) {
                                int p = bag[k];
                                if (features[p][featureID] < splitValue) {
                                    loss += customLoss(results[p] - meanLeft)  * weight[p];
                                } else {
                                    loss += customLoss(results[p] - meanRight) * weight[p];
                                }
                                if (loss > bestLoss) break; //OPTIONAL
                            }
                        }
                        
                        if (loss < bestLoss) {
                            bestLoss = loss;
                            bestValue = splitValue;
                            bestFeature = featureID;
                            bestLeft = totalLeft;
                            bestRight = totalRight;
                            if (loss == 0) goto outer;
                        }
                    }
                }
            }
        outer:
            
            if (bestLeft == 0 || bestRight == 0) {
                nodes[curNode].result = totalSum / totalWeight;
                continue;
            }
            
            if (config.computeImportances) {
                importances[bestFeature] += bRight - bLeft;
            }
            
            T mean = totalSum / totalWeight;
            
            T nextValue = -1e99;
            FOR(i, bLeft, bRight) if (features[bag[i]][bestFeature] < bestValue) nextValue = max(nextValue, features[bag[i]][bestFeature]);
            
            TreeNode left;
            TreeNode right;
            
            left.level = right.level = nodes[curNode].level + 1;
            nodes[curNode].feature = bestFeature;
            nodes[curNode].value = (bestValue + nextValue) / 2.0;
            if (!(nodes[curNode].value > nextValue)) nodes[curNode].value = bestValue;
            nodes[curNode].left = nodes.S;
            nodes[curNode].right = nodes.S + 1;
            
            int bMiddle = bRight;
            FOR(i, bLeft, bMiddle) {
                if (features[bag[i]][nodes[curNode].feature] >= nodes[curNode].value) {
                    swap(bag[i], bag[--bMiddle]);
                    i--;
                    continue;
                }
            }
            
            assert(bestLeft == bMiddle - bLeft);
            assert(bestRight == bRight - bMiddle);
            
            left.left = bLeft;
            left.right = bMiddle;
            right.left = bMiddle;
            right.right = bRight;
            
            stack.push(nodes.S);
            stack.push(nodes.S + 1);
            
            nodes.PB(left);
            nodes.PB(right);
            
        }
        
        nodes.shrink_to_fit();
    }
    
    template <class T> double predict(VC<T> &features) {
        TreeNode *pNode = &nodes[0];
        while (true) {
            if (pNode->feature < 0) return pNode->result;
            pNode = &nodes[features[pNode->feature] < pNode->value ? pNode->left : pNode->right];
        }
    }
};

RNG gRNG(1);

class TreeEnsemble {
public:
    
    VC<DecisionTree> trees;
    VD importances;
    MLConfig config;
    
    void clear() {
        trees.clear();
        trees.shrink_to_fit();
    }
    
    template <class T> DecisionTree createTree(VC<VC<T>> &features, VC<T> &results, MLConfig &config, int seed) {
        if (config.useBootstrapping) {
            if (config.lossFunction == MLConfig::MAE) {
                return DecisionTree(features, results, config, seed, true, MLConfig::MAE);
            } else if (config.lossFunction == MLConfig::MSE) {
                return DecisionTree(features, results, config, seed, true, MLConfig::MSE);
            } else if (config.lossFunction == MLConfig::MCE) {
                return DecisionTree(features, results, config, seed, true, MLConfig::MCE);
            } else if (config.lossFunction == MLConfig::CUSTOM) {
                return DecisionTree(features, results, config, seed, true, MLConfig::CUSTOM);
            }
        } else {
            if (config.lossFunction == MLConfig::MAE) {
                return DecisionTree(features, results, config, seed, false, MLConfig::MAE);
            } else if (config.lossFunction == MLConfig::MSE) {
                return DecisionTree(features, results, config, seed, false, MLConfig::MSE);
            } else if (config.lossFunction == MLConfig::MCE) {
                return DecisionTree(features, results, config, seed, false, MLConfig::MCE);
            } else if (config.lossFunction == MLConfig::CUSTOM) {
                return DecisionTree(features, results, config, seed, false, MLConfig::CUSTOM);
            }
        }
        // just to avoid compilation error
        return DecisionTree(features, results, config, seed, false, MLConfig::CUSTOM);
    }
    
    LL countTotalNodes() {
        LL rv = 0;
        REP(i, trees.S) rv += trees[i].nodes.S;
        return rv;
    }
    
    void printImportances() {
        assert(config.computeImportances);
        
        VC<pair<double, int>> vp;
        REP(i, importances.S) vp.PB(MP(importances[i], i));
        sort(vp.rbegin(), vp.rend());
        
        REP(i, importances.S) printf("#%02d: %.6lf\n", vp[i].Y, vp[i].X);
    }
    
};

class RandomForest : public TreeEnsemble {
public:
    
    template <class T> void train(VC<VC<T>> &features, VC<T> &results, MLConfig &_config, int treesMultiplier = 1) {
        double startTime = getTime();
        config = _config;
        
        int treesNo = treesMultiplier * config.treesNo;
        
        if (config.threadsNo == 1) {
            REP(i, treesNo) {
                if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
                trees.PB(createTree(features, results, config, gRNG.next()));
            }
        } else {
            thread *threads = new thread[config.threadsNo];
            mutex mutex;
            REP(i, config.threadsNo)
            threads[i] = thread([&] {
                while (true) {
                    mutex.lock();
                    int seed = gRNG.next();
                    mutex.unlock();
                    auto tree = createTree(features, results, config, seed);
                    mutex.lock();
                    if (trees.S < treesNo)
                        trees.PB(tree);
                    bool done = trees.S >= treesNo || config.timeLimit && getTime() - startTime > config.timeLimit;
                    mutex.unlock();
                    if (done) break;
                }
            });
            REP(i, config.threadsNo) threads[i].join();
            delete[] threads;
        }
        
        if (config.computeImportances) {
            importances = VD(features[0].S);
            for (DecisionTree tree : trees)
                REP(i, importances.S)
                importances[i] += tree.importances[i];
            double sum = 0;
            REP(i, importances.S) sum += importances[i];
            REP(i, importances.S) importances[i] /= sum;
        }
    }
    
    template <class T> double predict(VC<T> &features) {
        assert(trees.S);
        
        double sum = 0;
        REP(i, trees.S) sum += trees[i].predict(features);
        return sum / trees.S;
    }
    
    template <class T> VD predict(VC<VC<T>> &features) {
        assert(trees.S);
        
        int samplesNo = features.S;
        
        VD rv(samplesNo);
        if (config.threadsNo == 1) {
            REP(j, samplesNo) {
                REP(i, trees.S) rv[j] += trees[i].predict(features[j]);
                rv[j] /= trees.S;
            }
        } else {
            thread *threads = new thread[config.threadsNo];
            REP(i, config.threadsNo)
            threads[i] = thread([&](int offset) {
                for (int j = offset; j < samplesNo; j += config.threadsNo) {
                    REP(k, trees.S) rv[j] += trees[k].predict(features[j]);
                    rv[j] /= trees.S;
                }
            }, i);
            REP(i, config.threadsNo) threads[i].join();
            delete[] threads;
        }
        return rv;
    }
    
    template <class T> VC<T> predictOOB(VC<VC<T>> &features) {
        assert(config.saveChosenSamples);
        assert(trees.S);
        assert(features.S == trees[0].samplesChosen.S);
        
        VC<T> rv(features.S);
        VI no(features.S);
        
        for (auto tree : trees) REP(i, tree.samplesChosen.S) if (tree.samplesChosen[i] == 0) {
            rv[i] += tree.predict(features[i]);
            no[i]++;
        }
        
        REP(i, features.S) rv[i] /= max(1, no[i]);
        return rv;
    }
    
};

class BoostedForest : public TreeEnsemble {
public:
    
    VD currentResults;
    
    void clear() {
        trees.clear();
        trees.shrink_to_fit();
        currentResults.clear();
    }
    
    template <class T> void train(VC<VC<T>> &features, VC<T> &results, MLConfig &_config, int treesMultiplier = 1) {
        double startTime = getTime();
        config = _config;
        
        int treesNo = treesMultiplier * config.treesNo;
        
        if (currentResults.S == 0) currentResults = VD(results.S);
        
        if (config.threadsNo == 1) {
            VC<T> residuals(results.S);
            REP(i, treesNo) {
                if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
                REP(j, results.S) residuals[j] = results[j] - currentResults[j];
                trees.PB(createTree(features, residuals, config, gRNG.next()));
                REP(j, results.S) currentResults[j] += trees[trees.S-1].predict(features[j]) * config.shrinkage;
            }
        } else {
            //TODO: improve MT speed
            mutex mutex;
            for (int i = 0; i < treesNo; i += config.threadsNo) {
                if (config.timeLimit && getTime() - startTime > config.timeLimit) break;
                
                int usedThreads = min(config.threadsNo, treesNo - i);
                VC<T> residuals(results.S);
                REP(j, results.S) residuals[j] = results[j] - currentResults[j];
                
                thread *threads = new thread[config.threadsNo];
                REP(j, usedThreads)
                threads[j] = thread([&] {
                    mutex.lock();
                    int seed = gRNG.next();
                    mutex.unlock();
                    
                    auto tree = createTree(features, residuals, config, seed);
                    VD estimates(results.S);
                    REP(k, estimates.S) estimates[k] = tree.predict(features[k]) * config.shrinkage;
                    
                    mutex.lock();
                    trees.PB(tree);
                    REP(k, estimates.S) currentResults[k] += estimates[k];
                    mutex.unlock();
                });
                
                REP(j, usedThreads) threads[j].join();
                delete[] threads;
            }
        }
        
        if (config.computeImportances) {
            importances = VD(features[0].S);
            for (DecisionTree tree : trees)
                REP(i, importances.S)
                importances[i] += tree.importances[i];
            double sum = 0;
            REP(i, importances.S) sum += importances[i];
            REP(i, importances.S) importances[i] /= sum;
        }
    }
    
    template <class T> double predict(VC<T> &features, int treeLo = 0, int treeHi = -1) {
        assert(trees.S);
        if (treeHi == -1) treeHi = trees.S;
        
        double sum = 0;
        if (config.threadsNo == 1) {
            FOR(i, treeLo, treeHi) sum += trees[i].predict(features);
        } else {
            thread *threads = new thread[config.threadsNo];
            VD sums(config.threadsNo);
            int order = 0;
            REP(i, config.threadsNo)
            threads[i] = thread([&](int offset) {
                for (int j = treeLo + offset; j < treeHi; j += config.threadsNo)
                    sums[offset] += trees[j].predict(features);
            }, i);
            REP(i, config.threadsNo) threads[i].join();
            REP(i, config.threadsNo) sum += sums[i];
            delete[] threads;
        }
        return sum * config.shrinkage;
    }
};

struct Sample {
    VD data;
    int id;
    int source;
    double dist;
    double cycles;
    double complexity;
    double cargo;
    double stops;
    int start_day;
    double start_month;
    double start_day_of_month;
    double start_day_of_week;
    double start_time;
    double days;
    int pilot;
    int pilot2;
    double pilot_exp;
    double pilot_visits_prev;
    double pilot_hours_prev;
    double pilot_duty_hrs_prev;
    double pilot_dist_prev;
    double route_risk_1;
    double route_risk_2;
    double weather;
    double visibility;
    double traf0;
    double traf1;
    double traf2;
    double traf3;
    double traf4;
    double event0;
    double event1;
    double event2;
    double event3;
    double eventSum;
    bool training = false;
    
    Sample(string &s) {
        VS v = splt(s, ',');
        int pos = 0;
        
        data = VD(v.S, 0);
        REP(i, v.S) data[i] = atof(v[i].c_str());
        
        id = atoi(v[pos++].c_str());
        source = atoi(v[pos++].c_str());
        dist = atof(v[pos++].c_str());
        cycles = atof(v[pos++].c_str());
        complexity = atof(v[pos++].c_str());
        cargo = atof(v[pos++].c_str());
        stops = atof(v[pos++].c_str());
        start_day = atoi(v[pos++].c_str());
        start_month = atof(v[pos++].c_str());
        start_day_of_month = atof(v[pos++].c_str());
        start_day_of_week = atof(v[pos++].c_str());
        start_time = (v[pos][0]-'0')*10+(v[pos][1]-'0')+((v[pos][3]-'0')*10+(v[pos][4]-'0'))*0.01; pos++;
        days = atof(v[pos++].c_str());
        pilot = atoi(v[pos++].c_str());
        pilot2 = atoi(v[pos++].c_str());
        pilot_exp = atof(v[pos++].c_str());
        pilot_visits_prev = atof(v[pos++].c_str());
        pilot_hours_prev = atof(v[pos++].c_str());
        pilot_duty_hrs_prev = atof(v[pos++].c_str());
        pilot_dist_prev = atof(v[pos++].c_str());
        route_risk_1 = atof(v[pos++].c_str());
        route_risk_2 = atof(v[pos++].c_str());
        weather = atof(v[pos++].c_str());
        visibility = atof(v[pos++].c_str());
        traf0 = atof(v[pos++].c_str());
        traf1 = atof(v[pos++].c_str());
        traf2 = atof(v[pos++].c_str());
        traf3 = atof(v[pos++].c_str());
        traf4 = atof(v[pos++].c_str());
        
        if (v.S == 29) return;
        training = true;
        event0 = atof(v[pos++].c_str());
        event1 = atof(v[pos++].c_str());
        event2 = atof(v[pos++].c_str());
        event3 = atof(v[pos++].c_str());
        eventSum = atof(v[pos++].c_str());
    }
};

struct FieldType {
    VVD xevents;
    VVD xdistance;
    VVD pevents;
    VVD xsamples;
    VVD xcrash;
    VVD ycrash;
    VD events;
    VD distance;
    VD samples;
    
    FieldType() { }
    
    FieldType(int size) {
        events = VD(size, 0);
        distance = VD(size, 0);
        samples = VD(size, 0);
        pevents = VVD(size, VD(4, 0));
        xevents = VVD(size, VD(150, 0));
        xdistance = VVD(size, VD(150, 0));
        xsamples = VVD(size, VD(150, 0));
        xcrash = VVD(size, VD(150, -1000));
        ycrash = VVD(size, VD(150, +1000));
    }
    
    void add(int type, Sample &s) {
        double sum = 0;
        
        sum += min(MAX_EVENT, s.event0);
        sum += min(MAX_EVENT, s.event1);
        sum += min(MAX_EVENT, s.event2);
        sum += min(MAX_EVENT, s.event3);
        events[type] += sum;
        distance[type] += s.dist;
        samples[type]++;
        
        pevents[type][0] += s.event0;
        pevents[type][1] += s.event1;
        pevents[type][2] += s.event2;
        pevents[type][3] += s.event3;
        
        xevents[type][s.start_day] += sum;
        xdistance[type][s.start_day] += s.dist;
        xsamples[type][s.start_day]++;
        xcrash[type][s.start_day] = s.start_day;
        if (s.eventSum) ycrash[type][s.start_day] = s.start_day;
    }
    
    void finalize() {
        for (auto &v : xevents) FOR(i, 1, v.S) v[i] += v[i-1];
        for (auto &v : xdistance) FOR(i, 1, v.S) v[i] += v[i-1];
        for (auto &v : xsamples) FOR(i, 1, v.S) v[i] += v[i-1];
        for (auto &v : xcrash) FOR(i, 1, v.S) v[i] = max(v[i], v[i-1]);
        for (auto &v : ycrash) for (int i = v.S - 1; i >= 1; i--) v[i-1] = min(v[i], v[i-1]);
    }
};

const int MAX_PILOTS = 700;
const int MAX_SOURCES = 60;
const int MAX_WEATHERS = 10;
const int MAX_CARGOS = 10;
const int MAX_DAYS = 400;

FieldType pilot;
FieldType source;
FieldType weather;
FieldType cargo;
FieldType day;
FieldType hour;
FieldType weekday;


INLINE double safediv(double a, double b) {
    return b ? a / b : 0;
}

VD someWeights;

VD extractFeatures(Sample &s, bool randomize = false) {
    // VD rv;
    // REP(i, 29) rv.PB(s.data[i]);
    // rv.PB(s.route_risk_1-s.route_risk_2);
    // rv.PB(safediv(pilot.events[s.pilot], pilot.samples[s.pilot]));
    // rv.PB(safediv(pilot.events[s.pilot], pilot.distance[s.pilot]));
    // rv.PB(pilot.distance[s.pilot]);
    // rv.PB(safediv(source.events[s.source], source.samples[s.source]));
    // rv.PB(safediv(source.events[s.source], source.distance[s.source]));
    // rv.PB(safediv(cargo.events[s.cargo], cargo.samples[s.cargo]));
    // rv.PB(safediv(cargo.events[s.cargo], cargo.distance[s.cargo]));
    // rv.PB(s.traf0 + s.traf3 * 10 + s.traf2 * 30 + s.traf1 * 100);
    // rv.PB(safediv(day.events[s.start_day], day.samples[s.start_day]));
    // rv.PB(safediv(day.events[s.start_day], day.distance[s.start_day]));
    // rv.PB(safediv(weather.events[s.weather], weather.samples[s.weather]));
    // rv.PB(safediv(weather.events[s.weather], weather.distance[s.weather]));
    
    VD rv;
    FOR(i, 2, 29) if (i != 25) rv.PB(s.data[i]);
    rv.PB(s.route_risk_1-s.route_risk_2);
    rv.PB(safediv(pilot.events[s.pilot], pilot.samples[s.pilot]));
    // rv.PB(safediv(pilot.events[s.pilot], pilot.distance[s.pilot]));
    // rv.PB(safediv(pilot.distance[s.pilot], pilot.samples[s.pilot]));
    // rv.PB(safediv(s.dist, pilot.distance[s.pilot]));
    rv.PB(pilot.distance[s.pilot]);
    rv.PB(safediv(source.events[s.source], source.samples[s.source]));
    rv.PB(safediv(source.events[s.source], source.distance[s.source]));
    // rv.PB(source.distance[s.source]);
    rv.PB(safediv(cargo.events[s.cargo], cargo.samples[s.cargo]));
    rv.PB(safediv(cargo.events[s.cargo], cargo.distance[s.cargo]));
    rv.PB(cargo.distance[s.cargo]);
    rv.PB(s.traf0 * someWeights[0] + s.traf1 * someWeights[1] + s.traf2 * someWeights[2] + s.traf3 * someWeights[3] + s.traf4 * someWeights[4]);
    
    // rv.PB(safediv(day.events[s.start_day-1], day.distance[s.start_day-1]));
    // rv.PB(safediv(weather.events[s.weather], weather.samples[s.weather]));
    // rv.PB(safediv(weather.events[s.weather], weather.distance[s.weather]));
    
    
    // rv.PB(s.start_day - pilot.xcrash[s.pilot][s.start_day-1]);
    // rv.PB(safediv(pilot.xevents[s.pilot][s.start_day-1], pilot.xsamples[s.pilot][s.start_day-1]));
    // rv.PB(safediv(pilot.xevents[s.pilot][s.start_day-1], pilot.xdistance[s.pilot][s.start_day-1]));
    // rv.PB(safediv(source.xevents[s.source][s.start_day-1], source.xsamples[s.source][s.start_day-1]));
    // rv.PB(safediv(source.xevents[s.source][s.start_day-1], source.xdistance[s.source][s.start_day-1]));
    
    rv.PB(s.id % 10000);
    rv.PB(safediv(pilot.distance[s.pilot], pilot.samples[s.pilot]));
    rv.PB(safediv(source.distance[s.source], source.samples[s.source]));
    
    return rv;
}

VD extractFeaturesLR(Sample &s) {
    VD rv;
    // rv.PB(safediv(pilot.events[s.pilot], pilot.samples[s.pilot]));
    // rv.PB(safediv(pilot.events[s.pilot], pilot.distance[s.pilot]));
    // rv.PB(safediv(source.events[s.source], source.samples[s.source]));
    // rv.PB(safediv(source.events[s.source], source.distance[s.source]));
    // rv.PB(safediv(cargo.events[s.cargo], cargo.samples[s.cargo]));
    // rv.PB(safediv(cargo.events[s.cargo], cargo.distance[s.cargo]));
    rv.PB(s.pilot_exp);
    rv.PB(s.route_risk_1);
    rv.PB(s.route_risk_2);
    rv.PB(s.dist);
    rv.PB(s.cycles);
    return rv;
}

VD extractBonusFeatures(Sample &s) {
    VD rv;
    rv.PB(s.start_day - pilot.xcrash[s.pilot][s.start_day-1]);
    // rv.PB(safediv(s.dist, safediv(source.events[s.source], source.distance[s.source])));
    return rv;
}

VD genLev2Features(VD &v) {
    int N = v.S;
    
    VD rv;
    REP(i, N) rv.PB(v[i]);
    REP(i, N) REP(j, i) rv.PB(v[i] * v[j]);
    REP(i, N) REP(j, i) rv.PB(safediv(v[i], v[j]));
    return rv;
}

VVD genLev2Features(VVD &v) {
    VVD rv;
    REP(i, v.S) rv.PB(genLev2Features(v[i]));
    return rv;
}

VS genLev2FeaturesNames(VS &v) {
    int N = v.S;
    
    VS rv;
    REP(i, N) rv.PB(v[i]);
    REP(i, N) REP(j, i) rv.PB("(" + v[i] + ") * (" + v[j] + ")");
    REP(i, N) REP(j, i) rv.PB("(" + v[i] + ") / (" + v[j] + ")");
    return rv;
}

class Utils {
    
public:
    
    static VI generateSequence(int n, int start = 0, int step = 1) {
        VI rv(n);
        REP(i, n) rv[i] = start + i * step;
        return rv;
    }
    
    static VI generatePermutation(int n) {
        VI rv = generateSequence(n);
        REP(i, n) swap(rv[i], rv[i + rand() % (n - i)]);
        return rv;
    }
    
    static VI generateSubset(int n, int k) {
        VI v = generatePermutation(n);
        VI rv(k);
        REP(i, k) rv[i] = v[i];
        return rv;
    }
};

double calcScore(VD &pred, VD &truth) {
    assert(pred.S == truth.S);
    
    int N = 0;
    int M = 0;
    for (auto x : truth) {
        N += x >= 1;
        M += x >= 2;
    }
    
    VD points(pred.S, 0);
    VD bonusPoints(pred.S, 0);
    double maxScore = 0;
    REP(i, pred.S) {
        points[i] = max(0.0, 1.0 * (2 * N - i) / (2 * N));
        bonusPoints[i] = 0.3 * max(0.0, 1.0 * (2 * M - i) / (2 * M));
        if (i < N) maxScore += points[i];
        if (i < M) maxScore += bonusPoints[i];
    }
    
    VC<pair<double,int>> vp;
    REP(i, pred.S) vp.PB(MP(pred[i], i));
    sort(vp.rbegin(), vp.rend());
    VI rv(vp.S);
    REP(i, vp.S) rv[vp[i].Y] = i+1;
    
    double s0 = 0;
    double s1 = 0;
    REP(i, pred.S) {
        if (truth[i] >= 1) s0 += points[rv[i]-1];
        if (truth[i] >= 2) s1 += bonusPoints[rv[i]-1];
        // mse += (algo.predictions[i] - testResult[i]) * (algo.predictions[i] - testResult[i]);
    }
    s0 /= maxScore;
    s1 /= maxScore;
    s0 *= 1e6;
    s1 *= 1e6;
    return s0 + s1;	
}

double calcScore(VD &pred, VD &truth, int testSize, int runs) {
    RNG r(1);
    VI order;
    REP(i, pred.S) order.PB(i);
    double totalScore = 0;
    REP(run, runs) {
        REP(i, testSize) swap(order[i], order[r.next(i, order.S)]);
        VD xpred(testSize);
        VD xtruth(testSize);
        REP(i, testSize) {
            xpred[i] = pred[order[i]];
            xtruth[i] = truth[order[i]];
        }
        totalScore += calcScore(xpred, xtruth);
    }
    return totalScore / runs;
}

void removeColumns(VD &v, VI &col) {
    for (int j = col.S - 1; j >= 0; j--) v.erase(v.begin() + col[j]);
}

template <class T> VC<VC<T>> selectColumns(VC<VC<T>> &v, VI &col) {
    VC<VC<T>> rv(v.S, VC<T>(col.S));
    REP(j, v.S) REP(i, col.S) rv[j][i] = v[j][col[i]];
    return rv;
}

template <class T> VC<T> selectColumns(VC<T> &v, VI &col) {
    VC<T> rv(col.S);
    REP(i, col.S) rv[i] = v[col[i]];
    return rv;
}

template <class T> VC<VC<T>> multiply(VC<VC<T>> &A, VC<VC<T>> &B) {
    assert(A[0].S == B.S);
    VC<VC<T>> C(A.S, VC<T>(B[0].S, 0));
    REP(k, A[0].S) REP(i, A.S) REP(j, B[0].S) C[i][j] += A[i][k] * B[k][j];
    return C;
}

template <class T> VC<VC<T>> transpose(VC<VC<T>> &A) {
    VC<VC<T>> X(A[0].S, VC<T>(A.S, 0));
    REP(i, A.S) REP(j, A[0].S) X[j][i] = A[i][j];
    return X;
}

template <class T> VC<VC<T>> matrix(VC<T> &v) {
    VC<VC<T>> X(v.S, VC<T>(1));
    REP(i, v.S) X[i][0] = v[i];
    return X;
}

template <class T> VC<VC<T>> solveLeastSquares(VC<VC<T>> &A, VC<VC<T>> &B, double alpha = 0) {
    assert(A.S == B.S);
    
    VC<VC<T>> AT = transpose(A);
    VC<VC<T>> M0 = multiply(AT, A);
    int n = M0.S;
    REP(i, n) M0[i][i] += alpha;
    
    VC<VC<T>> L(n, VC<T>(n, 0));
    bool valid = true;
    
    REP(j, n) {
        double d = 0.0;
        REP(k, j) {
            double s = 0.0;
            REP(i, k) s += L[k][i] * L[j][i];
            s = (M0[j][k] - s) / L[k][k];
            L[j][k] = s;
            d += s * s;
        }
        d = M0[j][j] - d;
        // if (d <= 0) DB(d);
        if (d == 0) d = 1e-7;
        valid &= d > 0;
        L[j][j] = sqrt(max(d, 0.0));
        FOR(k, j + 1, n) L[j][k] = 0.0;
    }
    
    if (!valid) return VC<VC<T>>();
    
    VC<VC<T>> M1 = multiply(AT, B);
    int nx = B[0].S;
    
    REP(k, n) REP(j, nx) {
        REP(i, k) M1[k][j] -= M1[i][j] * L[k][i];
        M1[k][j] /= L[k][k];
    }
    
    for (int k = n - 1; k >= 0; k--) REP(j, nx) {
        FOR(i, k + 1, n) M1[k][j] -= M1[i][j] * L[i][k];
        M1[k][j] /= L[k][k];
    }
    
    return M1;
}

template <class T> VC<T> solveLeastSquares(VC<VC<T>> &A, VC<T> &b, double alpha = 0) {
    VC<VC<T>> B(b.S, VC<T>(1));
    REP(i, B.S) B[i][0] = b[i];
    VC<VC<T>> X = solveLeastSquares(A, B, alpha);
    if (X.S == 0) return VC<T>();
    VC<T> R(X.S);
    REP(i, R.S) R[i] = X[i][0];
    return R;
}

class LogisticRegression {
public:
    VD weights;
    
    template <class T> void train(VC<VC<T>> &features, VC<T> &results, double regularization = 0) {
        assert(features.S == results.S);
        weights = solveLeastSquares(features, results, regularization);
    }
    
    template <class T> double predict(VC<T> &features) {
        T rv = 0;
        REP(i, features.S) rv += features[i] * weights[i];
        return rv;
    }
    
};

void storeMatrixAsLibSVM(const char* fileName, const VVD &mat, const VD &Y) {
    FILE *fp;
    if (!(fp = fopen(fileName, "w"))) {
        throw runtime_error("Failed to open file!");
    }
    
    // write to the buffer
    for (int row = 0; row < mat.size(); row++) {
        // write class value first
        double val = Y[row];
        fprintf(fp, "%f", val);
        int index = 1;
        for (int col = 0; col < mat[row].size(); col++) {
            val = mat[row][col];
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

class TripSafetyFactors {
    
public:
    
    VD predictions;
    
    VI predict(VS &trainingData, VS &testingData) {
        double startTime = getTime();
        
        VVD trainData;
        VVD testData;
        VVD trainDataLR;
        VVD testDataLR;
        VD trainResults;
        VD trainResultsSimple;
        VD groundTruth;
        VD trainResultsType[4];
        
        pilot = FieldType(MAX_PILOTS);
        source = FieldType(MAX_SOURCES);
        weather = FieldType(MAX_WEATHERS);
        cargo = FieldType(MAX_CARGOS);
        day = FieldType(MAX_DAYS);
        hour = FieldType(MAX_DAYS);
        weekday = FieldType(MAX_DAYS);
        
        LogisticRegression LR;
        VVD LRData;
        
        for (string &r : trainingData) {
            Sample s(r);
            VD v;
            v.PB(s.traf0);
            v.PB(s.traf1);
            v.PB(s.traf2);
            v.PB(s.traf3);
            v.PB(s.traf4);
            trainResults.PB(s.eventSum);
            LRData.PB(v);
        }
        
        LR.train(LRData, trainResults, 0.01);
        someWeights = LR.weights;
        
        for (string &r : trainingData) {
            Sample s(r);
            pilot.add(s.pilot, s);
            source.add(s.source, s);
            cargo.add(s.cargo, s);
            weather.add(s.weather, s);
            day.add(s.start_day, s);
            hour.add((int)s.start_time, s);
            weekday.add(s.start_day_of_week, s);
        }
        
        pilot.finalize();
        source.finalize();
        weather.finalize();
        cargo.finalize();
        day.finalize();
        hour.finalize();
        weekday.finalize();
        
        for (string &r : trainingData) {
            Sample s(r);
            trainData.PB(extractFeatures(s));
            groundTruth.PB(s.eventSum);
            groundTruth.PB(min(1.0, s.eventSum));
            REP(i, 4) trainResultsType[i].PB(min(MAX_EVENT, s.data[29+i]));
        }
        
        for (string &r : testingData) {
            Sample s(r);
            testData.PB(extractFeatures(s));
        }
        
        MLConfig cfg;
        cfg.treesNo = 200;//24000;//
        cfg.randomFeatures = {1, 2, 2, 3, 3, 4};
        cfg.randomPositions = {2};
        cfg.maxNodeSize = 10;
        cfg.bagSize = 1.5;
        cfg.threadsNo = THREADS_NO;
        cfg.lossFunction = MLConfig::MAE;
        
        trainData = genLev2Features(trainData);
        testData  = genLev2Features(testData);
        
        storeMatrixAsLibSVM("/Users/yaric/train.libsvm", trainData, trainResults);
        
        
        VI columns;
        REP(i, 38) if (i != 1 && i != 4 && i != 7 && i != 8 && i != 15 && i != 18 && i != 19 && i != 21) columns.PB(i);
        columns.PB(559);
        columns.PB(1242);
        columns.PB(532);
        columns.PB(1401);
        columns.PB(1231);
        columns.PB(1286);
        
        print(trainData[0]);
        
        cerr << endl << trainData[0].size() << endl;
        
        trainData = selectColumns(trainData, columns);
        testData = selectColumns(testData, columns);
        
        print(trainData[0]);
        
        cerr << endl << trainData[0].size() << endl;
        
//        REP(i, trainingData.S) {
//            Sample s(trainingData[i]);
//            trainDataLR.PB(extractFeaturesLR(s));
//        }
//        
//        REP(i, testingData.S) {
//            Sample s(testingData[i]);
//            testDataLR.PB(extractFeaturesLR(s));
//        }
//        
//        LogisticRegression LRP;
//        LRP.train(trainDataLR, trainResults, 0.01);
        
        REP(i, trainingData.S) {
            Sample s(trainingData[i]);
            VD v = extractBonusFeatures(s);
            // v.PB(LRP.predict(trainDataLR[i]));
            REP(j, v.S) trainData[i].PB(v[j]);
        }
        
        REP(i, testingData.S) {
            Sample s(testingData[i]);
            VD v = extractBonusFeatures(s);
            // v.PB(LRP.predict(testDataLR[i]));
            REP(j, v.S) testData[i].PB(v[j]);
        }
        
        if (modeAll) {
            cfg.saveChosenSamples = true;
            cfg.treesNo *= 2;
        } else {
            cfg.timeLimit = 60 * 3;
        }
        
        RandomForest RF;
        RF.train(trainData, trainResults, cfg);
        
        if (modeAll) {
            VD oobpred = RF.predictOOB(trainData);
            
            double mse = 0;
            REP(i, trainData.S) mse += (oobpred[i] - groundTruth[i]) * (oobpred[i] - groundTruth[i]);
            cout << "Total Time: " << (getTime() - startTime) << endl;
            cout << "Predicted Score: " << calcScore(oobpred, groundTruth, (int)(groundTruth.S * (1 - SPLIT_SIZE)), 1000) << endl;
            cout << "MSE: " << sqrt(mse / trainData.S) << endl;
            exit(0);
        }
        
        VC<pair<double,int>> vp;
        predictions = VD(testData.S, 0);
        REP(i, testData.S) {
            predictions[i] = RF.predict(testData[i]);
            vp.PB(MP(predictions[i], i));
        }
        sort(vp.rbegin(), vp.rend());
        
        VI rv(vp.S);
        REP(i, vp.S) rv[vp[i].Y] = i+1;
        return rv;
    }
    
};

#endif
