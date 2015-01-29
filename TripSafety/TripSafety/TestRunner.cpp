//
//  TestRunner.cpp
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 1/16/15.
//  Copyright (c) 2015 yaric. All rights reserved.
//
#include <iostream>

#define FROM_TEST_RUNNER

#include "stdc++.h"
#include "TripSafetyFactors.cpp"

namespace am {
    namespace launcher {
        
        /*
         * The random class generator.
         */
        struct SecureRandom {
            int seed;
            
            SecureRandom(int seed) : seed(seed){}
            
            int nextInt(int max) {
                std::default_random_engine engine(seed);
                std::uniform_int_distribution<int> distribution(0, max - 1);
                
                return distribution(engine);
            }
        };
        
        SecureRandom rnd(1);
        
        VS splt(std::string s, char c = ',') {
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
        
        class TripSafetyFactorsVis {
            int numTrainingData;
            
            int X, Y;
            
            // the number of trips with one event
            int N;
            // the number of trips with than one event
            int M;

            VS DTrain;
            VS DTest;
            VI groundTruth;
            
            TripSafetyFactors *test;
        public:
            std::string dataFile = "../data/exampleData.csv";
            
            TripSafetyFactorsVis(TripSafetyFactors *tsf) : test(tsf) {}
            
            double doExec() {
                numTrainingData = 32967;
                
                if (!generateTest()) {
                    std::cerr << "Failed to generate test data" << std::endl;
                    return -1;
                }
                
                //
                // Start solution testing
                //
                VI results = test->predict(DTrain, DTest);
                
                //
                // Calculate score
                //
                Assert(results.size() == Y, "Wrong results count returned, expected: %i, found: %lu", Y, results.size());
                
                // check that all ranks unigue
                for (int i = 0; i < Y; i++) {
                    int rank = results[i];
                    for (int j = 0; j < Y; j++) {
                        if ( j != i && rank == results[j]) {
                            fprintf(stderr, "Duplicate rank found: %i at index: %i", rank, j);
                            return 0;
                        }
                    }
                }
                bool check[Y];
                for (int i = 0; i < Y; i++) {
                    if (results[i] < 1) {
                        fprintf(stderr, "Rank at index: %i, is lower than 1.", i);
                        return 0.0;
                    }
                    if (results[i] > Y) {
                        fprintf(stderr, "Rank at index: %i, is higher than the number of trips.", i);
                        return 0.0;
                    }
                    if(check[results[i] - 1]) {
                        fprintf(stderr, "Rank at index: %i, is not unique.", i);
                        return 0.0;
                    }
                    else {
                        check[results[i] - 1] = true;
                    }
                }
                
                // FILL points
                VD points(Y, 0);
                VD bonusPoints(Y, 0);
                
                double MaxPossibleScore = 0.0;
                for (int i = 0; i < Y; i++) {
                    if (N > 0) points[i] = std::max(0.0, (2.0 * N - i) / (2.0 * N));
                    if (M > 0) bonusPoints[i] = std::max(0.0, 0.3 * (2.0 * M - i) / (2.0 * M));
                    
                    if(i < N) MaxPossibleScore += points[i];
                    if(i < M) MaxPossibleScore += bonusPoints[i];
                }
                
                // calculate score
                double score = 0.0;
                for (int i = 0; i < Y; i++) {
                    if(groundTruth[i] > 0) {
                        score += points[results[i] - 1]; //true positive
                    }
                    if(groundTruth[i] > 1) {
                        score += bonusPoints[results[i] - 1]; //high safety concern
                    }
                }
                return score / MaxPossibleScore * 1000000;
            }
            
        private:
            
            /**
             * generates test data.
             *
             * return false in case of error.
             */
            bool generateTest() {
                X = numTrainingData * .66;
                Y = numTrainingData - X;
                
                fprintf(stderr, "Data file: %s, train size: %i, test size: %i\n", dataFile.c_str(), X, Y);
                
                
                //
                // load data
                //
                std::ifstream datafile (dataFile);
                if (!datafile.is_open()) {
                    std::cerr << "Error in opening file: " << dataFile << std::endl;
                    return false;
                }

                VS dataLines;
                std::string line;
                int row;
                for (row = 0; row < numTrainingData; row++) {
                    getline(datafile, line);
                    
                    dataLines.push_back(line);
                }
                datafile.close();
                
                // shuffle loaded data
                // obtain a time-based seed:
                unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                shuffle(dataLines.begin(), dataLines.end(), std::default_random_engine(seed));
                
                fprintf(stderr, "Loaded: %lu lines\n", dataLines.size());
                
                Assert(dataLines.size() == numTrainingData, "Expected: %i lines, but was: %lu", numTrainingData, dataLines.size());
                
                // parse data and create ground truth
                for (row = 0; row < numTrainingData; row++) {
                    VS items = splt(line, ',');
                    size_t items_size = items.size();
                    
                    if (row < X) {
                        // collect training data
                        DTrain.push_back(line);
                    } else {
                        // collect test data
                        DTest.push_back(line);
                        
                        // collect ground truth
                        int event = std::stoi(items[items_size - 1]);
                        if (event > 1) {
                            M++;
                        } else if (event > 0) {
                            N++;
                        }
                        groundTruth.push_back(event);
                    }
                }
                
                Assert(DTrain.size() == X, "Wrong train data size, expected: %i, found: %lu", X, DTrain.size());
                Assert(DTest.size() == Y, "Wrong test data size, expected: %i, found: %lu", Y, DTest.size());
                
                return true;
            }
            
        };
        
    }
}

int main(int argc, const char * argv[]) {
    if (argc < 1) {
        printf("Usage: dataFile\n");
        return 0;
    }
    TripSafetyFactors task;
    am::launcher::TripSafetyFactorsVis runner(&task);
    runner.dataFile = argv[1];
    
    int tests = 1;
    double mean = 0;
    
    for (int i = 0; i < tests; i++) {
        double score = runner.doExec();
        fprintf(stderr, "%i.) Score  = %f\n", i, score);
    }
    mean /= tests;
    fprintf(stderr, "Mean score: %f for %i tests\n", mean, tests);
    return 0;
}

