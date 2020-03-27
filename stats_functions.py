#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy.stats
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math, sys
import pandas as pd
import logging, logging.config


logger = logging.getLogger(__name__)

logging.basicConfig(level="DEBUG", filename="stats.log", filemode="a")

"""
    function b = bias_skill(predicted,reference)
%BIAS Calculate the bias between two variables (B)
%
%   B = BIAS_SKILL(PREDICTED,REFERENCE) calculates the bias between two variables 
%   PREDICTED and REFERENCE. The bias is calculated using the
%   formula:
%
%   B = mean(p) - mean(r)
%
%   where p is the predicted values, and r is the reference values. Note 
%   that p & r must have the same number of values.
%
%   Input:
%   PREDICTED : predicted field
%   REFERENCE : reference field
%
%   Output:
%   B : bias between predicted and reference

% Validate input args
    :return: 
"""
def biase_skill(reference, predicted):
    return numpy.mean(predicted) - numpy.mean(reference)


def rmse(obs, pred):
    """
    Root-mean-square error - Root Mean Squared Error is the average deviation of the predictions from the observations.

    Args:

    * obs: 1D array like - observed data

    * pred: 1D array like - predicated data

    Returns:
        float

    """
    if obs.shape != pred.shape:
        logger.erro('Number of data and estimate field dimensions do not match!')
        return None

    return numpy.sqrt(((numpy.array(obs) - numpy.array(pred)) ** 2).mean())


def sum_of_squares(data):
    """
    Calculate the sum of squares for an array like.

    Args:

    * data: array like

    Returns:
        float

    """
    return numpy.sum([x ** 2 for x in data])


def spread_data(data, scale):
    """
    Spread all the data points (from the mean) by the given scale.

    Args:

    * data: array like

    * scale: float

    Returns:
        array like

    """
    mean = numpy.mean(data)
    spread_data = []
    for val in data:
        spread_data.append(((val - mean) * scale) + mean)
    return spread_data



def shift_data(data, shift):
    """
    Shift all data points by given amount.

    Args:

    * data: array like

    * shift: float

    Returns:
        array like

    """
    return numpy.array(data) + shift



def blend_data(blend_data, fixed_data, blend):
    """
    Blend data towards fixed data using some crazy maths.

    Args:

    * blend_data: array like
        Data to be blended.

    * fixed_data: array like
        Data for which the blend_data is blended towards.

    * blend: float
        Percentage value of blend.

    Returns:
        array like

    """
    fcst_mean = numpy.mean(blend_data)
    fcst_std = numpy.std(blend_data)
    clim_mean = numpy.mean(fixed_data)
    xbar_of_blend = (((100. - blend) * fcst_mean) + (blend * clim_mean)) / 100.
    xbar_2n = (xbar_of_blend ** 2) * 100.
    sx_2f = ((sum_of_squares(blend_data) * (100. - blend)) / len(blend_data)) \
            + ((sum_of_squares(fixed_data) * blend) / len(fixed_data))
    stdv_of_blend = ((sx_2f - xbar_2n) / 100.) ** 0.5
    blended_data = []
    for val in blend_data:
        adjusted_val = (((val - fcst_mean) / fcst_std) * stdv_of_blend) + \
                       xbar_of_blend
        blended_data.append(adjusted_val)

    return blended_data


def gerrity_score(contingency_table, see_scoring_weights=False):
    """
    Calculate the Gerrity score for a given contingency table. The table
    columns must refer to observations and rows to forecasts.

    Args:

    * contingency_table: array like
        For the array to represent a vaild contingency table it must have 2
        dimensions only and each must have the same length, e.g. 3x3 or 4x4.

    Kwargs:

    * see_scoring_weights: boolean
        Set True to print the calculated scoring weights for the table.

    Returns:
        float

    """
    contingency_table = numpy.array(contingency_table, dtype=numpy.float64)
    # Check the table shape and allocate the number of rows and columns.
    tab_shape = contingency_table.shape
    assert len(tab_shape) == 2 and tab_shape[0] == tab_shape[1], \
        'Invalid contingency table. Table array must have 2 dimensions of ' \
        'the same size. Shape of this table: (%s, %s)' % (tab_shape[0],
                                                          tab_shape[1])
    rows_cols = tab_shape[0]

    total = numpy.sum(contingency_table)
    probs_table = contingency_table / total
    obs_distribution = numpy.sum(probs_table, axis=0)

    odds_ratios = []
    for j in range(rows_cols - 1):
        prob_sum = sum([obs_distribution[i] for i in range(j + 1)])
        odds_ratios.append((1 - prob_sum) / prob_sum)

    scoring_weights = numpy.zeros(tab_shape)
    for i in range(rows_cols):
        for j in range(rows_cols):
            if j >= i:
                comp1 = sum([1. / odds_ratios[x] for x in range(i)])
                comp2 = sum([odds_ratios[x] for x in range(j, rows_cols - 1)])
                comp3 = j - i
                scoring_weights[i, j] = (comp1 + comp2 - comp3) / \
                                        (rows_cols - 1.)
            else:
                scoring_weights[i, j] = scoring_weights[j, i]
    if see_scoring_weights:
        print(scoring_weights)
    return numpy.sum(scoring_weights * probs_table)


def spline(X_1d, Y_1d, s=0):
    """
    class scipy.interpolate.UnivariateSpline(x, y, w=None, bbox=[None, None], k=3, s=None, ext=0, check_finite=False)
    One-dimensional smoothing spline fit to a given set of data points.
    Fits a spline y = spl(x) of degree k to the provided x, y data. s specifies the number of knots by specifying a
    smoothing condition.

    :return: function
    """
    if len(X_1d) != len(Y_1d):
        print("Wrong format. TODO .............")
        return

    return UnivariateSpline(X_1d, Y_1d, s=s)



# q-Quantiles are values that partition a finite set of values into q subsets of (nearly) equal sizes
# There are q - 1 of the q-quantiles, one for each integer k satisfying 0 < k < q.
#
# For a finite population of N equally probable values indexed 1, ..., N from lowest to highest,
# the k-th q-quantile of this population can equivalently be computed via the value of Ip = N k/q.
#
# If Ip is not an integer, then round up to the next integer to get the appropriate index;
# the corresponding data value is the k-th q-quantile.
# If, instead of using integers k and q, the "p-quantile" is based on a real number p with 0 < p < 1
# then p replaces k/q in the above formula
#
# On the other hand, if Ip is an integer then any number from the data value at that index to the
# data value of the next can be taken as the quantile, and it is conventional (though arbitrary)
# to take the average of those two values
#
# 2-quantile(median), 3-quantiles(terciles), 4-quantiles(quartiles), 5-quantiles(quintiles), 100-quantiles(percentiles)
# Tercile each containing 33.333% of the data. The lower, and upper terciles are computed
# by ordering the data from smallest to largest and then finding the values below which
# fall 1/3 and 2/3 of the data.
# example:(0.097561, -1.158), (0.894309, 1.388), (0.195122, -0.9), (0.796748, 0.914), (0.333333, -0.633), (0.666667, 0.344), (-0.0905, 61.0)]
def q_quantile(data, k=4):

    # Note: this is same as "percentile_boundaries" method below
    pr = data.flatten()
    pr.sort()
    print(pr)
    sz = len(pr)

    quantile_arr = []
    for i in range(1,k):
        p = float(i) / float(k)
        # returns the smallest integer which is greater than or
        # equal to the input value (i.e. 30.5 ==> 31.0)
        rank = int(math.ceil(sz * p))
        quantile_arr.append((rank,pr[rank]))

    print(quantile_arr)
    return quantile_arr

    # Note: this is same as "percentile_boundaries" method below
    o_asc = numpy.reshape(data, data.size)
    o_asc.sort()
    print(o_asc)
    print(len(o_asc))

    # Given a vector V with N non masked values, the median of V is the middle
    # value of a sorted copy of V (Vs) - i.e. Vs[(N-1)/2], when N is odd, or
    # {Vs[N/2 - 1] + Vs[N/2]} / 2 when N is even.
    middle = numpy.median(o_asc)

    # calculate tercile
    sz = len(o_asc) + 1

    result = []
    for i in range(1, sz):
        percent = float(i) / float(sz)
        val = o_asc[i - 1]
        l = "%f,%f" %(percent,val)
        result.append(l)

    print(result)

    for r in result:
        u,v = r.split(",")
        if float(u) >= 0.1 :    # driest decile
            rank_arr.append((float(u), float(v)))
            break

    for r in result:
        u,v = r.split(",")
        if float(u) >= 0.9 :    # wettest decile
            rank_arr.append((float(u), float(v)))
            break

    for r in result:
        u, v = r.split(",")
        if float(u) >= 0.2 :    # driest quintile
            rank_arr.append((float(u), float(v)))
            break

    for r in result:
        u, v = r.split(",")
        if float(u) >= 0.8 :    # wettest quintile
            rank_arr.append((float(u), float(v)))
            break

    for r in result:
        u, v = r.split(",")
        if float(u) > 0.333 :     # driest tercile
            rank_arr.append((float(u), float(v)))
            break

    for r in result:
        u, v = r.split(",")
        if float(u) > 0.667 :     # wettest tercile
            rank_arr.append((float(u), float(v)))
            break

    rank_arr.append((middle, float(len(o_asc)/2)))
    print(rank_arr)
    return rank_arr


"""
SKILL_SCORE_MURPHY Calculate nondimensional skill score (SS) between two variables
%
%   SS = SKILL_SCORE_MURPHY(PREDICTED,REFERENCE) calculates the 
%   non-dimensional skill score (SS) difference between two variables 
%   PREDICTED and REFERENCE. The skill score is calculated using the
%   formula:
%
%   SS = 1 - RMSE^2/SDEV^2
%
%   where RMSE is the root-mean-squre error between the predicted and
%   reference values
%
%   (RMSD)^2 = sum_(n=1)^N (p_n - r_n)^2/N
%
%   and SDEV is the standard deviation of the reference values
%
%   SDEV^2 = sum_(n=1)^N [r_n - mean(r)]^2/(N-1)
%
%   where p is the predicted values, r is the reference values, and
%   N is the total number of values in p & r. Note that p & r must
%   have the same number of values.
%
%   Input:
%   PREDICTED : predicted field
%   REFERENCE : reference field
%
%   Output:
%   SS : skill score
%
%   Reference:
%   Allan H. Murphy, 1988: Skill Scores Based on the Mean Square Error and
%   Their Relationships to the Correlation Coefficient. Mon. Wea. Rev.,
%   116, 2417–2424. 
%   doi: http://dx.doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2
"""
def morphy_score(a, rmse):
    return 1 - numpy.square(rmse) / numpy.square(std(a))

"""
The standard deviation is the square root of the average of the squared 
deviations from the mean, i.e., std = sqrt(mean(abs(x - x.mean())**2)) . 
The average squared deviation is normally calculated as x.sum() / N , 
where N = len(x) . If, however, ddof is specified, the divisor N - ddof 
is used instead.
"""
def std(a):
    """
    numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<class 'numpy._globals._NoValue'>)[source]
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
    The standard deviation is computed for the flattened array by default, otherwise over the specified axis.

    a = np.array([[1, 2], [3, 4]])
    np.std(a)   1.1180339887498949
    np.std(a, axis=0)   array([ 1.,  1.])
    np.std(a, axis=1)   array([ 0.5,  0.5])
    """
    return numpy.std(a)

"""
Heidke Skill Score (HHS)
Compares how often the forecast category correctly match the observed category,
over the number of correct "hits" expected by chance alone.
This score utilizes the number of correct and incorrect category hits.
The values range from -50 to 100. A score of 100 indicates a perfect forecast and
a score of -50 indicates a perfectly incorrect forecast.
Scores greater than 0 indicate improvement compared to a random forecast and indicate skill.

HSS (%) = 100 * (H – E) / (T – E)  where:
H = Number of correct forecasts,
E = Expected number of correct forecasts (1/3 of total), and
T = Total number of valid forecast-observation pairs.
"""
def heidke_score(actual, predicted, bounds):
    # 2-quantile(median), 3-quantiles(terciles), 4-quantiles(quartiles), 5-quantiles(quintiles), 10-quantiles(deciles), 100-quantiles(percentiles)
    # quantile = [2, 3, 5, 10]
    #bounds = percentile_boundaries(actual, quantile)

    obs = value_category(actual, bounds)
    obs_data = numpy.array(obs).reshape(len(obs),1)
    dnn = value_category(predicted, bounds)
    dnn_data = numpy.array(dnn).reshape(len(dnn),1)

    boolarr = numpy.equal(obs_data,dnn_data)
    hits = numpy.sum(boolarr)

    E = len(predicted) / 3.0
    hss = 100 * (hits - E ) / ( len(predicted) - E )
    return (hss,boolarr, hits)




# create tables with stats
#    M       LT         UT        LQ      UQ      LD       UD
# [-0.0905, -0.625, 0.31733333, -0.835, 0.8998, -1.1444,  1.382]
def data_summary_mix(date, data):

    # y-axis data - rainfall
    actual = data[0]
    predicted = data[1]

    pd_data = numpy.concatenate((date, actual), axis = 1)
    pd_data = numpy.concatenate((pd_data, predicted), axis = 1)
    heading = ["Date", "Obsv", "Trn & Tst", "Observed", "DNN", "Observed", "DNN"]

    # 2-quantile(median), 3-quantiles(terciles), 4-quantiles(quartiles), 5-quantiles(quintiles), 10-quantile(decile), 100-quantiles(percentiles)
    perc = [2,3,5,10]
    bl = []
    for bound in perc:
        bounds = percentile_boundaries(actual, bound)
        for bnd in bounds:
            bl.append(bnd)

    print(bl)

    M = bl[0]
    LT = bl[1]
    UT = bl[2]
    LQ = bl[3]
    #print(bl[4])
    #print(bl[5])
    UQ = bl[6]
    LD = bl[7]
    # print(bl[8])
    # print(bl[9])
    # print(bl[10])
    # print(bl[11])
    # print(bl[12])
    # print(bl[13])
    # print(bl[14])
    UD = bl[15]

    stat_data = numpy.empty((len(date), 4), dtype="S15")  # 2 obs + 2 pred = 4
    stat_data.fill("               ")

    for i in range(len(actual)):
        if actual[i] <= LD:
            stat_data[i:i+1,0:1] = "extremely dry"
        elif actual[i] > LD and actual[i] <= LQ:
            stat_data[i:i+1,0:1] = "very dry"
        elif actual[i] > LQ and actual[i] <= LT:
            stat_data[i:i+1,0:1] = "dry"
        elif actual[i] > LT and actual[i] <= M:
            stat_data[i:i+1,0:1] = "normal to dry"

    i = 0
    for i in range(len(actual)):
        if actual[i] >= UD:
            stat_data[i:i+1,2:3] = "extremely wet"
        elif actual[i] < UD and actual[i] >= UQ:
            stat_data[i:i+1,2:3] = "very wet"
        elif actual[i] < UQ and actual[i] >= UT:
            stat_data[i:i+1,2:3] = "wet"
        elif actual[i] < UT and actual[i] >= M:
            stat_data[i:i+1,2:3] = "normal to wet"
    i = 0
    for i in range(len(predicted)):
        if predicted[i] <= LD:
            stat_data[i:i + 1, 1:2] = "extremely dry"
        elif predicted[i] > LD and predicted[i] <= LQ:
            stat_data[i:i + 1, 1:2] = "very dry"
        elif predicted[i] > LQ and predicted[i] <= LT:
            stat_data[i:i + 1, 1:2] = "dry"
        elif predicted[i] > LT and predicted[i] <= M:
            stat_data[i:i + 1, 1:2] = "normal to dry"
    i = 0
    for i in range(len(predicted)):
        if predicted[i] >= UD:
            stat_data[i:i + 1, 3:4] = "extremely wet"
        elif predicted[i] < UD and predicted[i] >= UQ:
            stat_data[i:i + 1, 3:4] = "very wet"
        elif predicted[i] < UQ and predicted[i] >= UT:
            stat_data[i:i + 1, 3:4] = "wet"
        elif predicted[i] < UT and predicted[i] >= M:
            stat_data[i:i + 1, 3:4] = "normal to wet"

    stat_array = numpy.concatenate((pd_data, stat_data), axis=1)
    q_class = [(LD,"Driest Decile"), (LQ,"Deriest Quintile"), (LT,"Deriest Tercile"), (M,"Median"), (UT,"wettest Tercile"), (UQ,"Wettest Quintile"), (UD,"Wettest Decile")]
    return (heading,stat_array, q_class)




def percentile_boundaries(data, num_of_categories):
    """
    Return the boundary values which split the given data set into the
    requested number of categories. E.g. data = [1,2,3,4] split into 3
    categories would return [2.0, 3.0] as the tercile boundaries.

    Args:

    * data: array like

    * num_of_categories: integer
        The number of categories wanted. Note, the function will always return
        num_of_categories - 1 values.

    Returns:
        list

    """

    percentiles = numpy.linspace(0, 100, num_of_categories + 1)[1:-1]

    bounds = [round(numpy.percentile(data, percentile), 8)
              for percentile in percentiles]

    return bounds

#                                     1                                                                                     2
# [('Median', -0.0945), ('Lowest Tercile', -0.61), ('Driest Quintile', -0.8032), ('Driest Decile', -1.1513), ('Upper Tercile', 0.31466667), ('Wettest Quintile', 0.8988), ('Wettest Decile', 1.382)]
def value_category(values, bounds, boundary_val_cat='outer',
                   middle_val_cat='upper'):
    """
    Given a set of values and boundaries, return each value's category.
    Categories are named numerically starting from 1. There are always
    1 + number of bounds categories.

    Args:

    * values: float or list of floats

    * bounds: list
        A list of boundary values. These are automatically sorted into numeric
        order.

    Kwargs:

    * boundary_val_cat:
        If a value equals a boundary value, specify whether it is placed in an
        inner or outer category. Default is outer.

    * middle_val_cat:
        If a value equals the middle boundary value (only for odd number of
        boundaries), specify whether it is placed in the upper or lower
        category. Default is upper.

    Returns:
        list

    """
    if boundary_val_cat not in ['inner', 'outer']:
        raise ValueError('%s is not a valid input, use "inner" or "outer"'
                         % boundary_val_cat)
    if middle_val_cat not in ['upper', 'lower']:
        raise ValueError('%s is not a valid input, use "upper" or "lower"'
                         % middle_val_cat)
    if not hasattr(values, '__iter__'):
        values = [values]
    bounds.sort()

    num_of_bounds = len(bounds)
    middle_index = float(num_of_bounds - 1) / 2.

    categories = []
    for value in values:
        category_found = False
        for index, bound in enumerate(bounds):
            if value > bound:
                continue
            elif value < bound:
                category_found = True
                categories.append(index + 1)
                break
            else:
                # When value equals a bound.
                if index < middle_index:
                    if boundary_val_cat == 'inner':
                        category = index + 2
                    else:
                        category = index + 1
                    category_found = True
                    categories.append(category)
                    break
                elif index > middle_index:
                    if boundary_val_cat == 'inner':
                        category = index + 1
                    else:
                        category = index + 2
                    category_found = True
                    categories.append(category)
                    break
                else:
                    # When value equals the middle bound.
                    if middle_val_cat == 'lower':
                        category = index + 1
                    else:
                        category = index + 2
                    category_found = True
                    categories.append(category)
                    break
        if not category_found:
            # The value is above all boundaries.
            categories.append(index + 2)
    # print(categories)
    return categories



def category_probabilities(values, bounds, boundary_val_cat='outer',
                           middle_val_cat='upper', return_counts=False):
    """
    Given a set of values and boundaries, return the associated probabilities
    for each category. There are always 1 + number of bounds categories.

    Args:

    * values: list
        A list of values.

    * bounds: list
        A list of boundary values. These are automatically sorted into numeric
        order.

    Kwargs:

    * boundary_val_cat:
        If a value equals a boundary value, specify whether it is placed in an
        inner or outer category. Default is outer.

    * middle_val_cat:
        If a value equals the middle boundary value (only for odd number of
        boundaries), specify whether it is placed in the upper or lower
        category. Default is upper.

    Returns:
        list

    """
    category_counts = [0.] * (len(bounds) + 1)
    num_of_vals = float(len(values))
    categories = value_category(values, bounds, boundary_val_cat,
                                middle_val_cat)
    for category in categories:
        category_counts[category - 1] += 1

    if return_counts:
        return [int(val) for val in category_counts]
    else:
        category_probs = [val / num_of_vals for val in category_counts]
        return category_probs


def skill_score(accuracy_score, reference_score, perfect_score):
    """
    Calculate the skill score.

    Args:

    * accuracy_score: float

    * reference_score: float

    * perfect_score: float

    Returns:
        float

    """
    skill_score = (accuracy_score - reference_score) / \
                  (perfect_score - reference_score)
    return skill_score


def pdf_probabilities(pdf, bounds):
    """
    Calculate the area of the PDF in between each bound, hence the probability.

    Args:

    * pdf: instance of scipy.stats.gaussian_kde

    * bounds: list
        A list of boundary values. These are automatically sorted into numeric
        order.

    """
    bounds.sort()
    extended_boundaries = [-numpy.inf] + bounds
    extended_boundaries.append(numpy.inf)
    probs = []
    for i in range(len(extended_boundaries) - 1):
        probs.append(pdf.integrate_box_1d(extended_boundaries[i], extended_boundaries[i + 1]))
    return probs


def pdf_percentile_boundaries(pdf, num_of_categories, accuracy_factor=50):
    """
    Estimate the boundary values when splitting a PDF in to equally sized
    areas.

    Args:

    * pdf: instance of scipy.stats.gaussian_kde

    * num_of_categories: integer
        The number of equally sized areas the PDF is split into.

    Kwargs:

    * accuracy_factor: integer
        The estimation is calculated using iteration, this value specifies how
        many values to split the PDF into and iterate over. Therefore, the
        higher the factor, the longer the calculation takes but the more
        accurate the returned values. Default is 50.

    Returns:
        list of bounds

    """
    dmin = numpy.min(pdf.dataset)
    dmax = numpy.max(pdf.dataset)
    x_vals = numpy.linspace(dmin, dmax, accuracy_factor)

    required_area_size = 1. / float(num_of_categories)
    bounds = []
    lower_bound = -numpy.inf
    for i, x_val in enumerate(x_vals):
        this_area_size = pdf.integrate_box_1d(lower_bound, x_val)
        if this_area_size > required_area_size:
            upper_diff = this_area_size - required_area_size
            lower_diff = required_area_size - \
                         pdf.integrate_box_1d(lower_bound, x_vals[i - 1])
            total_diff = upper_diff + lower_diff
            proportion_diff = upper_diff / total_diff

            val_diff = x_val - x_vals[i - 1]
            proportion_val_diff = val_diff * proportion_diff
            adjusted_x_val = x_val - proportion_val_diff
            bounds.append(adjusted_x_val)
            if len(bounds) == num_of_categories - 1:
                break
            lower_bound = adjusted_x_val

    return bounds


def calculate_pdf_limits(pdf, levels=50, range_limiter=20):
    """
    Calculate the values where the PDF stops. The range_limiter determines the
    value at which to cut the PDF outer limits. It is a proportional value not
    an actual value. The larger the given value the further out the extremes
    will be returned.

    Args:

    * pdf: instance of scipy.stats.gaussian_kde

    Kwargs:

    * levels : integer
        This determines the step size when calculating the limits.

    * range_limiter: scalar
        This value is used to calculate the range of the PDF. A PDF function
        can take a while to converge to 0, so to calculate sensible stop and
        start points, some proportional value above 0 is calculated. The given
        range_limiter value is used as factor to determine what that above 0
        value is. Simply, the higher the given value the wider the PDF limits.
        See nested function calculate_pdf_limits for more details.

    """
    dmin = numpy.min(pdf.dataset)
    dmax = numpy.max(pdf.dataset)
    pdf_min = numpy.mean([pdf(dmin)[0], pdf(dmax)[0]]) / float(range_limiter)
    # First calculate the appropriate step size given the data range and number
    # of levels.
    step_size = (dmax - dmin) / float(levels)
    while pdf(dmin)[0] > pdf_min:
        dmin -= step_size
    while pdf(dmax)[0] > pdf_min:
        dmax += step_size
    return dmin, dmax


def generate_pdf_values(data, levels=50, range_limiter=20,
                        bandwidth='silverman', return_pdf=False):
    """
    Calculate the PDF function and return a set of values and points along the
    curve.

    Args:

    * data : 1D array like
        List of data values.

    Kwargs:

    * levels : integer
        This determines how many points are returned. If plotting, higher
        values lead to smoother plots.

    * range_limiter: scalar
        This value is used to calculate the range of the PDF. A PDF function
        can take a while to converge to 0, so to calculate sensible stop and
        start points, some proportional value above 0 is calculated. The given
        range_limiter value is used as factor to determine what that above 0
        value is. Simply, the higher the given value the wider the PDF limits.
        See nested function calculate_pdf_limits for more details.

    * bandwidth: string, scalar or callable
        The method used to calculate the estimator bandwidth. This can be
        'scott', 'silverman', a scalar constant or a callable. If a scalar,
        this will be used directly as kernel-density estimate (kde) factor. If
        a callable, it should take a scipy.stats.gaussian_kde instance as only
        parameter and return a scalar. Default is 'silverman'.

    * return_pdf: boolean
        If True, the callable scipy.stats.gaussian_kde instance is also
        returned.

    Returns:
        PDF values, PDF points, PDF function (optional)

    """
    # Generate kernel density estimate (PDF)
    pdf = scipy.stats.gaussian_kde(data, bw_method=bandwidth)
    dmin, dmax = calculate_pdf_limits(pdf, levels, range_limiter)
    pdf_points = numpy.linspace(dmin, dmax, levels)
    if return_pdf:
        return pdf(pdf_points), pdf_points, pdf
    else:
        return pdf(pdf_points), pdf_points


def array_correlation(x, y, method='pearson'):
    """
    Calculate the correlation between each matching element of the arrays in x
    and y. Note, x and y must be identical in shape.

    Args:

    * x and y: List of arrays

    Kwargs:

    * method: 'pearson' or 'spearman'
        Pearson's correlation or Spearman's ranked correlation.

    Returns:
        Array of correlation values for corresponding elements.

    """
    assert method in ['pearson', 'spearman'], 'Invalid method %s.' % method
    x = numpy.ma.masked_array(x)
    y = numpy.ma.masked_array(y)
    assert x.shape[0] == y.shape[0], 'x and y must contain the same number of' \
                                     ' arrays.'
    assert x.shape[1:] == y.shape[1:], 'All arrays in x and y must be the ' \
                                       'same shape.'

    if method == 'pearson':
        corr_method = scipy.stats.pearsonr
    elif method == 'spearman':
        corr_method = scipy.stats.spearmanr

    correlation_array = numpy.empty(x.shape[1:])
    for index in numpy.ndindex(x.shape[1:]):
        array_index = tuple([slice(None)] + list(index))
        element_corr = corr_method(x[array_index], y[array_index])[0]
        correlation_array[index] = element_corr

    return numpy.ma.masked_invalid(correlation_array)


class ProbabilityAccuracyScores(object):
    """
    Class containing accuracy score methods for probabilistic methods. Lists
    for observations and probabilities must be given in the same order as each
    other. I.e. ob_categories[x] must refer to category_probs[x].

    Args:

    * ob_categories: integer or list of integers
        The numbers of the observed categories, 1 referring to the 1st. E.g.
        for tercile categories, 1 = lower, 2 = middle, 3 = upper.

    * category_probs: list
        A list of the probabilities for each category. The sum must equal 1
        when rounded to 5 decimal places, so irrational numbers need at least
        6 decimal places, e.g. 0.333333.

    """

    def __init__(self, ob_categories, category_probs):
        ob_categories = numpy.array(ob_categories)
        category_probs = numpy.array(category_probs)
        if category_probs.dtype == object:
            raise ValueError('Not all sets of probabilities contain the same' \
                             ' number of categories: %s' % category_probs)
        if category_probs.ndim > 2:
            raise ValueError('Too many dimensions in category_probs.')
        if (ob_categories.ndim + 1) != category_probs.ndim:
            raise ValueError('Must be exactly 1 observation for each set of ' \
                             'probabilities.')
        if ob_categories.ndim == 0:
            ob_categories = numpy.array([ob_categories])
            category_probs = numpy.array([category_probs])
        if ob_categories.shape[0] != category_probs.shape[0]:
            raise ValueError('Number of observations, %s, does not match the ' \
                             'number of probability sets, %s' % (
                                 ob_categories.shape[0], category_probs.shape[0]))
        if numpy.max(ob_categories) > category_probs.shape[1] or \
                        numpy.min(ob_categories) < 1:
            raise ValueError('Observation category, %s, out of range.' %
                             numpy.max(ob_categories))
        prob_totals = [round(val, 5)
                       for val in numpy.sum(category_probs, axis=1)]
        if prob_totals != [1] * category_probs.shape[0]:
            raise ValueError('All probability sets must sum to 1.')

        self.ob_categories = ob_categories
        self.category_probs = category_probs
        self.categories = range(1, category_probs.shape[1] + 1)
        self.num_of_categories = len(self.categories)

    def _ROC_plot(self, all_hit_rates, all_false_alarm_rates, ROC_scores,
                  title, categoriy_names, colours, save, show, axes):
        """
        Plot the ROC curves.

        """
        self._plot('ROC', all_false_alarm_rates, all_hit_rates, ROC_scores,
                   title, 'False alarm rate', 'Hit rate', 'ROC scores',
                   categoriy_names, colours, save, show, axes)

    def _reliability_plot(self, all_fcst_probs, all_obs_freqs, biases,
                          title, categoriy_names, colours, save, show, axes):
        """
        Plot the reliability diagram.

        """
        self._plot('Reliability', all_fcst_probs, all_obs_freqs, biases, title,
                   'Forecast probabilities', 'Observed frequency', 'Biases',
                   categoriy_names, colours, save, show, axes)

    def _plot(self, plot_type, all_x_vals, all_y_vals, scores, title, xlab,
              ylab, legend_title, categoriy_names, colours, save, show, axes):
        """
        Plot the specified plot type.

        """
        cmap = LinearSegmentedColormap.from_list('cmap', colours)
        colour_index = [int(round(val))
                        for val in numpy.linspace(0, 256,
                                                  len(categoriy_names))]
        if axes is None:
            plt.figure()
            ax = plt.axes()
        else:
            ax = axes
        legend_labels = []
        for i, (x_vals, y_vals, score, category) in enumerate(
                zip(all_x_vals,
                    all_y_vals,
                    scores,
                    categoriy_names)):
            if x_vals is not None and y_vals is not None:
                ax.plot(x_vals, y_vals, 'o-',
                        color=cmap(colour_index[i]))
                label = '{cat} = {score}'.format(cat=category,
                                                 score=round(score,
                                                             3))
                legend_labels.append(label)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if title is not None:
            plt.title(title)
        else:
            plt.title('%s Plot' % plot_type)
        plt.legend(tuple(legend_labels), 'lower right', fontsize='x-small',
                   title=legend_title)
        ax.plot([0, 1], [0, 1], 'k--')
        plt.grid()
        if save:
            plt.savefig(save)
        if show:
            plt.show()

    def _brier_score(self, ob_categories, category_probs):
        """
        Method to do the Brier score calculations.

        Returns:
            float

        """
        num_of_trails = len(category_probs)
        cumalative_score = 0.
        for ob_category, probability_set in zip(ob_categories,
                                                category_probs):
            for category, category_prob in zip(self.categories,
                                               probability_set):
                ob_value = 0.
                if category == ob_category:
                    ob_value = 1.
                cumalative_score += (category_prob - ob_value) ** 2
        brier_score = (1. / (2. * num_of_trails)) \
                      * cumalative_score
        return brier_score

    def _ranked_probability_score(self, ob_categories, category_probs):
        """
        Method to do the RPS calculations.

        Returns:
            float

        """
        num_of_trails = len(category_probs)
        cumalative_score = 0.
        for ob_category, probability_set in zip(ob_categories,
                                                category_probs):
            ob_cumalative = 0.
            prob_cumalative = 0.
            for category, category_prob in zip(self.categories,
                                               probability_set):
                if category == ob_category:
                    ob_cumalative += 1.
                prob_cumalative += category_prob
                cumalative_score += (prob_cumalative - ob_cumalative) ** 2
        ranked_prob_score = 1. - (1. /
                                  ((self.num_of_categories - 1.) \
                                   * num_of_trails) \
                                  * cumalative_score)
        return ranked_prob_score

    def _calculate_score(self, score_method, split_categories):
        """
        Run the given method, either for each category or for all combined.

        """
        if split_categories:
            scores = []
            for category in self.categories:
                category_indx = numpy.where(self.ob_categories == category)
                if len(category_indx[0]) > 0:
                    scores.append(round(score_method(
                        self.ob_categories[category_indx],
                        self.category_probs[category_indx]), 8))
                else:
                    scores.append(None)
            return scores
        else:
            return round(score_method(self.ob_categories, self.category_probs),
                         8)

    def brier_score(self, split_categories=True):
        """
        Used for unordered categories, e.g. rain or no rain.

        Kwargs:

        * split_categories: boolean
            If True, the score is calculated separately for each category.
            Otherwise a single value is returned derived from all categories.
            Default is True.

        Returns:
            list or float

        """
        return self._calculate_score(self._brier_score, split_categories)

    def probability_score(self, split_categories=True):
        """
        The negative orientation of the Brier score, i.e. 1 = perfect instead
        of 0.

        Kwargs:

        * split_categories: boolean
            If True, the score is calculated separately for each category.
            Otherwise a single value is returned derived from all categories.
            Default is True.

        Returns:
            list or float

        """
        if split_categories:
            return [1 - score for score in self.brier_score(split_categories)]
        else:
            return 1. - self.brier_score(split_categories)

    def ranked_probability_score(self, split_categories=True):
        """
        Used for ordered categories, e.g. tercile categories.


        Kwargs:

        * split_categories: boolean
            If True, the score is calculated separately for each category.
            Otherwise a single value is returned derived from all categories.
            Default is True.

        Returns:
            list or float

        """
        return self._calculate_score(self._ranked_probability_score,
                                     split_categories)

    def ROC_score(self, num_of_thresholds=6, outer_categories_only=False,
                  plot=False, title=None, category_names=None,
                  colours=['blue', 'green', 'red'], save=None, show=False,
                  axes=None):
        """
        Calculate the relative operating characteristic score for each
        category.

        Kwargs:

        * num_of_thresholds: integer
            The number of thresholds between 0 and 1 at which to compare the
            probabilities. Adjusting the number of thresholds is used to smooth
            the ROC curves. Higher numbers make less difference for smaller
            data sets and add on computing time. Adjusting can also affect the
            ROC scores, so it is recommended to use a consistent number when
            comparing results. Default is 6 i.e. [0, .2, .4, .6, .8, 1.]

        * outer_categories_only: boolean
            If True, only the ROC scores for the lowest and highest categories
            are calculated (and plotted if required).

        * plot: boolean
            Set True to plot and show the ROC curves. Default is False.

        Note, all kwargs from here are only used if plot=True

        * title: string
            Specify a title for the plot.

        * category_names: list
            Provide a list of category names. There must be the same number of
            names as there are categories and they categories are labelled from
            lowest to highest.

        * colours: list
            List of the colours from which to create a colour map for each
            category plot.

        * save: string
            Specify a file path string to save the plot.

        * show: boolean
            Specify whether to show the plot.

        * axes: matplotlib.pyplot.axes instance
            If given, the plot if done on this axes.

        Returns:
            list of ROC scores for each category.

        """
        thresholds = numpy.linspace(0., 1., num_of_thresholds)
        if outer_categories_only:
            categories = [self.categories[0], self.categories[-1]]
        else:
            categories = self.categories
        all_hit_rates = []
        all_false_alarm_rates = []
        ROC_scores = []
        for category in categories:
            occurances = (self.ob_categories == category).sum()
            if occurances == 0:
                all_hit_rates.append(None)
                all_false_alarm_rates.append(None)
                ROC_scores.append(None)
                continue
            non_occurances = len(self.ob_categories) - occurances
            if non_occurances == 0:
                raise UserWarning('ROC scores cannot be calculated when all ' \
                                  'observations are in the same category.')
            hit_rates = []
            false_alarm_rates = []
            for threshold in thresholds:
                hits = 0.
                false_alarms = 0.
                for ob_category, probability_set in zip(self.ob_categories,
                                                        self.category_probs):
                    # Look at the relevant category only.
                    category_prob = probability_set[category - 1]
                    if category_prob >= threshold:
                        if ob_category == category:
                            hits += 1.
                        else:
                            false_alarms += 1.
                hit_rates.append(hits / float(occurances))
                false_alarm_rates.append(false_alarms / float(non_occurances))
            ROC_score = 0.
            for i in xrange(num_of_thresholds - 1):
                # Calculate the area under each bit of curve using trapezium
                # rule.
                ROC_score += ((hit_rates[i] + hit_rates[i + 1]) / 2.) * \
                             (false_alarm_rates[i] - false_alarm_rates[i + 1])
            all_hit_rates.append(hit_rates)
            all_false_alarm_rates.append(false_alarm_rates)
            ROC_scores.append(ROC_score)
        if plot:
            if category_names:
                assert len(category_names) == len(categories), '%s ' \
                                                               'category_names must be provided.' % len(categories)
            else:
                category_names = ['Category %s' % cat for cat in categories]
            self._ROC_plot(all_hit_rates, all_false_alarm_rates, ROC_scores,
                           title, category_names, colours, save, show, axes)
        return ROC_scores

    def reliability(self, num_of_bins=10, outer_categories_only=False,
                    plot=False, title=None, category_names=None,
                    colours=['blue', 'green', 'red'], save=None, show=False,
                    axes=None):
        """
        Calculate the reliability of the forecasts. This is best assessed with
        a reliability plot (set plot to True). The overall forecasting biases
        are returned, values under and over 1 represent under and over
        confidence respectively.

        Kwargs:

        * num_of_bins: integer
            The number of bins to split 0 to 1 into. Forecast probabilities are
            then placed into the corresponding bins.

        * outer_categories_only: boolean
            If True, only the reliability for the lowest and highest categories
            are calculated (and plotted if required).

        * plot: boolean
            Set True to plot and show the reliability diagram. Default is
            False.

        Note, all kwargs from here are only used if plot=True

        * title: string
            Specify a title for the plot.

        * category_names: list
            Provide a list of category names. There must be the same number of
            names as there are categories and they categories are labelled from
            lowest to highest.

        * colours: list
            List of the colours from which to create a colour map for each
            category plot.

        * save: string
            Specify a file path string to save the plot.

        * show: boolean
            Specify whether to show the plot.

        * axes: matplotlib.pyplot.axes instance
            If given, the plot if done on this axes.

        Returns:
            list of biases for each category.


        """
        bin_bounds = numpy.linspace(0., 1., num_of_bins + 1)[1:-1]
        if outer_categories_only:
            categories = [self.categories[0], self.categories[-1]]
        else:
            categories = self.categories
        all_fcst_probs = []
        all_obs_freqs = []
        biases = []
        for category in categories:
            # Get all forecasted probabilities for this category and assign
            # each to a bin.
            this_category_probs = self.category_probs[:, category - 1]
            fcst_prob_bins = value_category(this_category_probs, bin_bounds)

            fcst_probs = []
            obs_freqs = []
            for bin_category in range(1, num_of_bins + 1):
                # For each bin, collect all forecast probabilities which fell
                # into that bin and their associated observation.
                bin_fcst_probs = []
                obs_in_cat = []
                for prob, fcst_bin, ob_cat in zip(this_category_probs,
                                                  fcst_prob_bins,
                                                  self.ob_categories):
                    if fcst_bin == bin_category:
                        bin_fcst_probs.append(prob)
                        if ob_cat == category:
                            obs_in_cat.append(1.)
                        else:
                            obs_in_cat.append(0.)
                fcst_probs.append(numpy.mean(bin_fcst_probs))
                obs_freqs.append(numpy.mean(obs_in_cat))

            fcst_probs = numpy.ma.masked_invalid(fcst_probs)
            obs_freqs = numpy.ma.masked_invalid(obs_freqs)

            mean_fcst_prob = numpy.mean(fcst_probs)
            mean_obs_freq = numpy.mean(obs_freqs)
            bias = mean_fcst_prob / mean_obs_freq
            biases.append(bias)
            all_fcst_probs.append(fcst_probs)
            all_obs_freqs.append(obs_freqs)

        if plot:
            if category_names:
                assert len(category_names) == len(categories), '%s ' \
                                                               'category_names must be provided.' % len(categories)
            else:
                category_names = ['Category %s' % cat for cat in categories]
            self._reliability_plot(all_fcst_probs, all_obs_freqs, biases,
                                   title, category_names, colours, save, show,
                                   axes)
        return biases


class ArrayRegression(object):
    """
    Given a series of values (predictor) and a series of arrays (predictand) of
    the same length, calculate the regression equations for each point.

    Args:

    * predictor: 1D array or list

    * predictand: array like
        The first dimension must be the same length as the predictor.

    Kwargs:

    * num_of_polynomials: integer
        Specify the number of polynomials in the calculated regression
        equations.

    """

    def __init__(self, predictor, predictand, num_of_polynomials=1):
        predictor = numpy.array(predictor)
        predictand = numpy.ma.masked_array(predictand)
        if predictor.ndim != 1:
            raise UserWarning('Predictor must be 1 dimensional.')
        if len(predictor) <= 1:
            raise UserWarning('Predictor must contain at least 2 values.')
        if predictor.shape[0] != predictand.shape[0]:
            raise UserWarning('Predictor and predictand do not have the same ' \
                              'first dimension size. %s %s' % (
                                  predictor.shape,
                                  predictand.shape))
        self.predictor = predictor
        self.predictand = predictand
        # For numpy.polyfit to work with masked arrays, all masked values must
        # be replaced with numpy.nan.
        if self.predictand.mask.any() == True:
            self.predictand[self.predictand.mask] = numpy.nan

        self.regression_array = numpy.empty(self.predictand.shape[1:],
                                            dtype=numpy.object)
        # Break up the predictand array into 2 dimensional arrays which numpy
        # can handle.
        if self.predictand.ndim > 2:
            for latter_indicies in numpy.ndindex(self.predictand.shape[2:]):
                index = tuple([slice(None), slice(None)] +
                              list(latter_indicies))

                regress_coeffs = numpy.polyfit(self.predictor,
                                               self.predictand[index],
                                               num_of_polynomials)
                for i, coeffs in enumerate(regress_coeffs.transpose()):
                    regression_equation = numpy.poly1d(coeffs)
                    self.regression_array[index[1:]][i] = regression_equation
        else:
            regress_coeffs = numpy.polyfit(self.predictor, self.predictand,
                                           num_of_polynomials)
            for i, coeffs in enumerate(regress_coeffs.transpose()):
                regression_equation = numpy.poly1d(coeffs)
                self.regression_array[i] = regression_equation

    def __repr__(self):
        def repr_func(poly):
            poly_str = ''
            for coeff, order in zip(poly.coeffs, range(poly.order, -1, -1)):
                if poly.order != order:
                    if coeff < 0:
                        poly_str += ' - '
                    else:
                        poly_str += ' + '
                else:
                    if coeff < 0:
                        poly_str += ' -'
                poly_str += '%s' % round(abs(coeff), 2)
                if order > 0:
                    poly_str += 'x'
                    if order >= 2:
                        poly_str += '^%s' % order
            return poly_str

        vfunc = numpy.vectorize(repr_func)
        repr_arr = vfunc(self.regression_array)
        return '%s' % repr_arr

    def __call__(self, value):
        """
        Generate an array of values by evaluating each regression equation with
        a specified value.

        Args:

        * value: float
            Value for which to evaluate each regression equation.

        Returns
            numpy array

        """

        def evaluate_regression(reg_equation, value):
            return reg_equation(value)

        vfunc = numpy.vectorize(evaluate_regression)
        result = numpy.ma.masked_array(vfunc(self.regression_array, value))
        result = numpy.ma.masked_invalid(result)
        return result


class ArrayCategoryProbabilities(object):
    """
    Class for calculating probability values across arrays.

    Args:

    * value_array: list or array of arrays
        A set of arrays of values

    * bounds_arrays: list or array of arrays
        A set of arrays of bounds

    Kwargs:

    * boundary_val_cat:
        If a value equals a boundary value, specify whether it is placed in an
        inner or outer category. Default is outer.

    * middle_val_cat:
        If a value equals the middle boundary value (only for odd number of
        boundaries), specify whether it is placed in the upper or lower
        category. Default is upper.

    """

    def __init__(self, value_array, bounds_arrays, boundary_val_cat='outer',
                 middle_val_cat='upper'):
        bounds_arrays = [numpy.array(arr) for arr in bounds_arrays]
        array_shape = bounds_arrays[0].shape
        value_array = numpy.array(value_array)
        array_shape_err_message = 'Array shapes do not match.'
        for array in bounds_arrays:
            assert array.shape == array_shape, array_shape_err_message
        assert value_array.shape[1:] == array_shape, array_shape_err_message

        # For indexing the value array during calculation.
        self.indices = list(numpy.ndindex(array_shape))
        # numpy.vectorize runs the function once to check so start index on -1.
        self.indices_index = -1

        self.value_array = value_array
        self.bounds_arrays = bounds_arrays
        self.boundary_val_cat = boundary_val_cat
        self.middle_val_cat = middle_val_cat

    def caluculate(self):
        """
        Run the calculation of the probabilities for each array point.

        Returns:
            tuple of probability arrays, starting with lowest category

        """

        def calculate_probs(*bounds):
            index = tuple([slice(None)] + \
                          list(self.indices[self.indices_index]))
            self.indices_index += 1
            return tuple(category_probabilities(self.value_array[index],
                                                list(bounds),
                                                self.boundary_val_cat,
                                                self.middle_val_cat))

        vfunc = numpy.vectorize(calculate_probs)
        return vfunc(*self.bounds_arrays)


def array_category_probabilities(value_array, bounds_arrays,
                                 boundary_val_cat='outer',
                                 middle_val_cat='upper'):
    """
    Given a set of value arrays and a set of bounds arrays, calculate the
    probabilities for each category (defined by the bounds) at each point by
    counting the number of values within each of the categories. E.g. Given
    10 realisations of an x-y grid and 2 sets of bounds on the same x-y grid,
    return 3 x-y grids of probabilities, these being probabilities of, beneath
    the lower bounds, between the bounds, and above the upper bounds. For each
    array point the probabilities will add up to 1.

    Args:

    * value_array: list or array of arrays
        A set of arrays of values

    * bounds_arrays: list or array of arrays
        A set of arrays of bounds

    Kwargs:

    * boundary_val_cat:
        If a value equals a boundary value, specify whether it is placed in an
        inner or outer category. Default is outer.

    * middle_val_cat:
        If a value equals the middle boundary value (only for odd number of
        boundaries), specify whether it is placed in the upper or lower
        category. Default is upper.

    Returns:
        tuple of probability arrays, starting with lowest category

    """
    return ArrayCategoryProbabilities(value_array, bounds_arrays,
                                      boundary_val_cat,
                                      middle_val_cat).caluculate()
