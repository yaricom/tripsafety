//
//  statistics.h
//  TripSafety
//
//  Created by Iaroslav Omelianenko on 1/31/15.
//  Copyright (c) 2015 Danfoss. All rights reserved.
//

#ifndef TripSafety_statistics_h
#define TripSafety_statistics_h

static const double SQRTH  =  7.07106781186547524401E-1;
static const double MAXLOG =  7.09782712893383996732E2;

double errorFunction(double x);

/**
 * Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
 * Evaluates polynomial when coefficient of N is 1.0.
 * Otherwise same as <tt>polevl()</tt>.
 * <pre>
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 * </pre>
 * The function <tt>p1evl()</tt> assumes that <tt>coef[N] = 1.0</tt> and is
 * omitted from the array.  Its calling arguments are
 * otherwise the same as <tt>polevl()</tt>.
 * <p>
 * In the interest of speed, there are no checks for out of bounds arithmetic.
 *
 * @param x argument to the polynomial.
 * @param coef the coefficients of the polynomial.
 * @param N the degree of the polynomial.
 */
double p1evl( double x, double coef[], int N ) {
    
    double ans;
    ans = x + coef[0];
    
    for(int i = 1; i < N; i++) ans = ans * x + coef[i];
    
    return ans;
}

/**
 * Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
 * <pre>
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 * </pre>
 * In the interest of speed, there are no checks for out of bounds arithmetic.
 *
 * @param x argument to the polynomial.
 * @param coef the coefficients of the polynomial.
 * @param N the degree of the polynomial.
 */
double polevl( double x, double coef[], int N ) {
    
    double ans;
    ans = coef[0];
    
    for(int i = 1; i <= N; i++) ans = ans * x + coef[i];
    
    return ans;
}

/**
 * Returns the complementary Error function of the normal distribution.
 * <pre>
 *  1 - erf(x) =
 *
 *                           inf.
 *                             -
 *                  2         | |          2
 *   erfc(x)  =  --------     |    exp( - t  ) dt
 *               sqrt(pi)   | |
 *                           -
 *                            x
 * </pre>
 * <b>Implementation:</b>
 * For small x, <tt>erfc(x) = 1 - erf(x)</tt>; otherwise rational
 * approximations are computed.
 * <p>
 * Code adapted from the <A HREF="http://www.sci.usq.edu.au/staff/leighb/graph/Top.html">
 * Java 2D Graph Package 2.4</A>,
 * which in turn is a port from the
 * <A HREF="http://people.ne.mediaone.net/moshier/index.html#Cephes">Cephes 2.2</A>
 * Math Library (C).
 *
 * @param a the argument to the function.
 */
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

/**
 * Returns the error function of the normal distribution.
 * The integral is
 * <pre>
 *                           x
 *                            -
 *                 2         | |          2
 *   erf(x)  =  --------     |    exp( - t  ) dt.
 *              sqrt(pi)   | |
 *                          -
 *                           0
 * </pre>
 * <b>Implementation:</b>
 * For <tt>0 <= |x| < 1, erf(x) = x * P4(x**2)/Q5(x**2)</tt>; otherwise
 * <tt>erf(x) = 1 - erfc(x)</tt>.
 * <p>
 * Code adapted from the <A HREF="http://www.sci.usq.edu.au/staff/leighb/graph/Top.html">
 * Java 2D Graph Package 2.4</A>,
 * which in turn is a port from the
 * <A HREF="http://people.ne.mediaone.net/moshier/index.html#Cephes">Cephes 2.2</A>
 * Math Library (C).
 *
 * @param a the argument to the function.
 */
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

/**
 * Returns the area under the Normal (Gaussian) probability density
 * function, integrated from minus infinity to <tt>x</tt>
 * (assumes mean is zero, variance is one).
 * <pre>
 *                            x
 *                             -
 *                   1        | |          2
 *  normal(x)  = ---------    |    exp( - t /2 ) dt
 *               sqrt(2pi)  | |
 *                           -
 *                          -inf.
 *
 *             =  ( 1 + erf(z) ) / 2
 *             =  erfc(z) / 2
 * </pre>
 * where <tt>z = x/sqrt(2)</tt>.
 * Computation is via the functions <tt>errorFunction</tt> and <tt>errorFunctionComplement</tt>.
 *
 * @param a the z-value
 * @return the probability of the z value according to the normal pdf
 */
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

#endif
