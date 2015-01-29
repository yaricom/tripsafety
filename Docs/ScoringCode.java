            boolean[] check = new boolean[rv.length];
            for (int i = 0; i < rv.length; i++) {
                if (rv[i] < 1) {
                    lt.addFatalError("Rank at index " + i + " is lower than 1.");
                    return 0.0;
                }
                if (rv[i] > rv.length) {
                    lt.addFatalError("Rank at index " + i + " is higher than the number of trips.");
                    return 0.0;
                }
                if(check[rv[i] - 1]) {
                    lt.addFatalError("Rank at index " + i + " is not unique.");
                    return 0.0;
                }
                else {
                    check[rv[i] - 1] = true;
                }
            }

            double[] points = new double[rv.length];
            double[] bonusPoints = new double[rv.length];

            double MaxPossibleScore = 0.0;

            //FILL POINTS
            for (int i = 0; i < rv.length; i++) {
                if (N > 0) points[i] = Math.max(0, (2 * N - i) / (2.0 * N));
                if (M > 0) bonusPoints[i] = Math.max(0, 0.3 * (2 * M - i) / (2.0 * M));

                if(i < N) MaxPossibleScore += points[i];
                if(i < M) MaxPossibleScore += bonusPoints[i];
            }

            double score = 0.0;
            for (int i = 0; i < rv.length; i++) {
                if(groundTruth[i] > 0) {
                    score += points[rv[i] - 1]; //true positive
                }
                if(groundTruth[i] > 1) {
                    score += bonusPoints[rv[i] - 1]; //high safety concern
                }
            }
            return score / MaxPossibleScore * SCORE_MULTIPLIER;
			