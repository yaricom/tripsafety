//
//  ridge_regression.h
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 2/10/15.
//  Copyright (c) 2015 nologin. All rights reserved.
//

#ifndef TripSafety_ridge_regression_h
#define TripSafety_ridge_regression_h

/*  
 The data is read from tab-separated ASCI file rg.txt; each column is
 saved into auxilary files rg_xx.txt.
 
 File rg.txt: data set; column 0 = dependent var
 Files rg_xx.txt are copies of individual columns
 File rg_err.txt: residual error per observation
 File rg_log.txt: program logs: this file contains all results (regression
 coefficients, empirical distribution for regression coefficients, etc.)
 
 The structure of rg_log.txt easily allows for parsing and further
 statistical processing:
 
 col 1: xxxxx_yy where xxxxx is the name of the procedure being run
 and yy is the iteration.
 col 2: iteration in the Regression call; set VERBOSE to -1 if you
 don't want this level of detail - then only results obtained
 at the final iteration are saved in rg_log.txt
 col 3: variable optimized at current iteration
 col 4: lambda (internal variable)
 col 5: reduction in standard deviation of error achieved at current
 iteration; this value is between 0 and 1; 0 corresponds to all
 regression coefficients set to zero; 1 means perfect fit
 cols 6, 7, 8 etc.: current value of regression coefficients
 
 *param: regression coefficients
 *param_seed: initial regression coefficients (usually 0)
 obs: number of observations
 var: number of columns in input file
 MSE: mean squared error
 MSE_init: MSE when regression coefficients are 0
 init: flag to indicate that Regress_init has been called.
 seed: flag to indicate that Init_param has been called. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VERBOSE 0  /* 0 for detailed logs, -1 for summary */

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

long init=-1;
long obs=-1;
long var=-1;
long seed=-1;
double MSE=0;
double MSE_init=0;
double *param;
double *param_seed;

int main() {
    
    double deltaTime;
    time_t start,finish; /* to meaure time elapsed */
    
    system("del c:\\ftp\\IFDindex\\programs\\rg_*.txt");
    
    time(&start);
    
    Regress_init();
    Validation(2, 10, 150);
    Init_param(2, 5, 4);
    Bootstrap(20, 0, 100);
    Regress(3, 100, "REGRESSION", 1);
    
    time(&finish);
    deltaTime=difftime(finish,start);
    printf("Time elapsed %lf sec.\n",deltaTime);
    
    if (seed==1) { free(param_seed); }
    if (init==1) { free(param); }
    
    return 0;
}
    
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
        }
    }
    
    /**
     * Performs nvalid approximate regression with different starting
     * points to identify an initial approximate solution. This
     * solution will be used as starting point (seed) for the
     * regression.
     *
     * Input: mode: should be 2 (recommended) or 3; see Regression for
     * details
     * Input: nvalid: number of regressions to perform
     * Input: niter: number of iterations to use in
     * each regression; should be small here (<10)
     * Input: mode: should be 2 (recommended) or 3; see Regression for
     * description
     * Output (global): seed is set to 1 to indicate
     * that Regression must use param_seed
     * Output: param_seed: regression coefficients to be used as
     * starting point in Regression
     */
    void initFeaturesSeed(const VVD &train, const VD &check, const long mode, const long nvalid, const long niter) {
        //Init_param
        
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
     * detect outliers. The results (regression coefficients and reduction
     * in standard deviation of error for each of the nsample regressions)
     * are stored in rg_log.txt.
     *
     * Input (global): var, obs, ini (must be set to 1 by using Regress_init
     * first)
     * Input: niter: number of iterations to use to compute regression
     * coefficients; see Regression.
     * Input: nsample: number of samples; Boostrap performs one regression
     * on each sample
     * Input: mode: see Regression for details.
     * Output (global): all output in rg_log.txt
     */
    void bootstrap(const VVD &train, const VD &check, const long nsample, const long mode, const long niter) {
        VVD bootTrain;
        VD bootCheck;
        
        VVD bootTest;
        VD bootCheckTest;
        
        // the best found regression coefficients
        VD bestCoef(var, 0);
        
        
        // array to store selected indices
        long *pick;
        int k, idx, m, oi;
        double oobMSE, oobME, minOOBMSE = 1e+9;
        for (int n = 0; n < nsample; n++) {
            // pick up random observations
            for (k = 0; k < obs; k++) { pick[k]=0; }
            for (k = 0; k < obs; k++) {
                idx = rand() % obs;
                pick[idx]++;
            }
            
            // create subsample
            for (m = 0; m < obs; m++) {
                if (pick[m] > 0) {
                    // save pick[m] copies of row in data
                    for (k = 0; k < pick[m]; k++) {
                        bootTrain.push_back(train[k]);
                        bootCheck.push_back(check[k]);
                    }
                } else {
                    // save current row as test sample
                    bootTest.push_back(train[k]);
                    bootCheckTest.push_back(check[k]);
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
            oobMSE /= (double)bootTest.size();
            oobErrors.push_back(oobMSE);
            if (oobMSE < minOOBMSE) {
                oobMSE = minOOBMSE;
                // store current regression coefficients
                bestCoef.swap(regrCoef);
            }
        }
        
        // store best regression coefficients as train results
        regrCoef.swap(bestCoef);
    }
    
    
    double regress(const VVD &train, const VD &check, const long mode, const long niter, long seed_flag) {
        
        
        return 0;
    }
    
    double predict(const VD &test) {
        
        return 0;
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
    


/*---------------------------------------------------------------------*/

double Regress(long mode,long niter, char *label,long seed_flag){
    
    /*  
     Performs regression. Must run Regress_init first to compute obs,
     var, initialize init and allocate memory to param. The Regress function
     returns the reduction in standard deviation of error; this value is
     between 0 and 1; 0 corresponds to all regression coefficients set to
     zero; 1 means perfect fit. Regress does not compute confidence
     intervals for the coefficients (use Boostrap for this purpose).
     Regression coefficients and reduction in standard deviation of error
     are stored in rg_log.txt.
     
     Input (global): var, obs, seed, init
     Input: seed_flag: if 1 use initial regression coefficients param_seed
     to start iterative regression procedure; if seed_flag = 1 then
     Param_init must be called first to intialize  param_seed and seed.
     Input: niter: number of iterations to use to compute regression
     coefficients; if convergence is slow or erratic then increase
     niter and perform validation tests with Validation.
     Input: label: usually the name of the parent procedure calling
     Regression(e.g. Bootstrap, Validation); the label appears in
     rg_log.txt
     Input: mode: determines the type of algorithm used for regression:
     mode = 0: visits each variable sequentially starting with first
     variable; useful when variables are pre-sorted in such a
     way that the first few variables explain most of the
     variance.
     mode = 1: visits variables in random order; should be the default
     mode
     mode = 2: same as mode = 1
     mode = 3: visits variables in random order; in addition perform
     partial instead of full optimization on each variable
     (similar to simulated annealing to avoid getting stuck
     in a local optimun; drawback: slows convergence, may
     require to increase niter); useful when performing
     validations with Validation or when finding initial
     regression coefficients with Init_param
     
     Returns: reduction in standard deviation of error.
     Output (global): param: regression coefficients */
    
    long k,l,i,iter;
    double xd,sp,lambda,val,y,e_new,resvar;
    double *col,*err;
    char stri[80];
    char filename[1024];
    
    FILE *ERR,*RG0,*RG,*COL,*OUT_ERR,*LOG;
    
    stri[79]='\0';
    filename[1023]='\0';
    
    col=(double *)calloc(obs,2*sizeof(double *));  /* why 2? */
    err=(double *)calloc(obs,2*sizeof(double *));
    
    /* need to run Regress_init first if the data set is new */
    
    if (init != 1) {  printf("Must run Regress_init() first."); exit(1); }
    
    LOG=fopen("c:\\ftp\\IFDindex\\programs\\rg_log.txt","at");
    MSE_init=0;
    RG0=fopen("c:\\ftp\\IFDindex\\programs\\rg_0.txt","rt");
    ERR=fopen("c:\\ftp\\IFDindex\\programs\\rg_err.txt","wt");
    
    k=0;
    while (!feof(RG0)) {
        fscanf(RG0,"%s\n",stri);
        val=atof(stri);
        MSE_init+=val * val;
        err[k]=val;
        k++;
        fprintf(ERR,"%lf\n",val);
    }
    fclose(ERR);
    fclose(RG0);
    MSE_init=sqrt(MSE_init/obs);
    for (k=1; k<var; k++) { param[k]=0; }
    
    /* if seed=1 uses initial regressors */
    
    if (seed_flag==1) {
        if (seed !=1) { printf("Must run Init_param() first.\n"); exit(2); }
        for (k=1; k<var;k++) { param[k]=param_seed[k]; }
        RG=fopen("c:\\ftp\\IFDindex\\programs\\rg.txt","rt");
        ERR=fopen("c:\\ftp\\IFDindex\\programs\\rg_err.txt","wt");
        
        k=0;
        e_new=0;
        while (!feof(RG)) {
            fscanf(RG,"%s\n",stri);
            val=atof(stri);
            if (k>0) { e_new+=param[k]*val; } /* k=0 is the dependent var */
            if (k==0) { y=val; }
            k++;
            if (k==var) {  k=0; fprintf(ERR,"%lf\n",y-e_new); e_new=0; }
        }
        fclose(ERR);
        fclose(RG);
        printf("\n");
    }
    
    /* regression */
    
    for (iter=0; iter< niter; iter++) {
        
        if ((mode == 0)||(mode ==3)) {
            l= 1 + (iter % (var - 1));
        } else {
            l = 1 + (long)rand()%(var - 1);
        }
        
        Create_rgfilename(l,filename);
        COL=fopen(filename,"rt");
        
        k=0;
        while (!feof(COL)) {
            fscanf(COL,"%s\n",stri);
            val=atof(stri);
            col[k]=val;
            k++;
        }
        fclose(COL);
        
        
        ERR=fopen("c:\\ftp\\IFDindex\\programs\\rg_err.txt","rt");
        k=0;
        while (!feof(ERR)) {
            fscanf(ERR,"%s\n",stri);
            val=atof(stri);
            err[k]=val;
            k++;
        }
        fclose(ERR);
        
        xd=0; sp=0;
        for (k=0; k<obs; k++) {
            xd+=col[k]*col[k];
            sp+=col[k]*err[k];
        }
        if (xd==0) { printf("Empty column.\n"); exit(2); }
        lambda = sp/xd;
        if (mode==1) { lambda = lambda * rand()/(double)RAND_MAX; }
        
        
        OUT_ERR=fopen("c:\\ftp\\IFDindex\\programs\\rg_err.txt","wt");
        MSE=0;
        for (k=0; k<obs; k++) {
            e_new = err[k] - lambda * col[k];
            MSE+= e_new * e_new;
            fprintf(OUT_ERR,"%lf\n",e_new);
        }
        fclose(OUT_ERR);
        param[l]+=lambda;
        
        /* save results, compute resvar */
        
        if ((iter % 10 == VERBOSE)||(iter == niter-1)) {
            MSE = sqrt(MSE/obs);
            resvar = 1-MSE/MSE_init;
            if (strcmp(label,"NOPRINT")!=0) {
                fprintf(LOG,"%s\t%d\t%d\t%f\t%f",label,iter,l,lambda,resvar);
                for (k=1; k<var; k++) { fprintf(LOG,"\t%lf",param[k]); }
                fprintf(LOG,"\n");
                printf("REGRESS %ld\t%lf\n",iter,resvar);
            }
        }
    }
    
    fclose(LOG);
    free(err);
    free(col);
    return(resvar);
    
}

/*---------------------------------------------------------------------*/

int Create_rgfilename(long k, char *filename) {
    /* create filename associated with var k */
    char digits[5];
    
    digits[4]='\0';
    
    itoa(k,digits,10);
    strcpy(filename,"c:\\ftp\\IFDindex\\programs\\rg_");
    strcat(filename,digits);
    strcat(filename,".txt");
}

};
#endif
