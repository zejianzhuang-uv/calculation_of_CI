
using DataFrames
import Statistics
import Distributions
using PrettyTables



"""
Calaculate a confidence interval of a sample in 1, 2 -sigma confidence level

Reference:

1. https://scikit-hep.org/iminuit/notebooks/error_bands.html

2. https://www.geo.fu-berlin.de/en/v/soga-py/Basics-of-statistics/Inferential-Statistics/Interval-Estimate/index.html

3. https://rowannicholls.github.io/python/statistics/confidence_intervals.html

4. https://rowannicholls.github.io/python/statistics/agreement/coefficient_of_variation.html

5. https://datascience.stackexchange.com/questions/124648/understand-and-compute-confidence-interval-and-coefficient-of-variation-for-regr
"""
function confidence_interval(data::Vector{Float64}; k=1.5, cl=68e-2)
    filter_data = IQR_outlier_detection(data, k=k)
    # mean = Statistics.mean(filter_data)
    std = Statistics.std(filter_data)
    # cl : z
    # 0.68: 1 
    # 0.95:  1.96
    alpha = 1. - cl
    z = Statistics.quantile(Distributions.Normal(), 1- alpha/2)
    # Calculate margin of error
    merr = z * std
    return merr
end


function confidence_interval(data::AbstractDataFrame; k=1.5, cl=68e-2, formatters="%.1f", pretty_print=true)
    name = names(data)
    merr_vec = Float64[]
    for nn in name
        merr = confidence_interval(data[!, nn], k=k, cl=cl) 
        push!(merr_vec, merr)
    end
    df = DataFrame(name = name, merr = merr_vec)
    if pretty_print == true
        pretty_table(df, formatters=ft_printf(formatters), title="Confidenc level: $cl", title_alignment = :c, header_crayon = crayon"yellow bold")
    end
    return df
end


"""
Refernece:
https://rpubs.com/JuliaWorkshop/Confidence_Intervals_Julia
"""
function mean_confidence_interval(data::Vector{Float64}; k=1.5, cl=68e-2)
    filter_data = IQR_outlier_detection(data, k=k)
    # mean = Statistics.mean(filter_data)
    std = Statistics.std(filter_data)

    alpha = 1. - cl
    z = Statistics.quantile(Distributions.Normal(), 1- alpha/2)
    # Calculate margin of error
    merr = z * (std / sqrt(length(filter_data) ) )
    return merr
end

function mean_confidence_interval(data::AbstractDataFrame; k=1.5, cl=68e-2, formatters="%.3f", pretty_print=true)
    name = names(data)
    merr_vec = Float64[]
    for nn in name
        merr = mean_confidence_interval(data[!, nn], k=k, cl=cl) 
        push!(merr_vec, merr)
    end
    df = DataFrame(name = name, merr = merr_vec)
    if pretty_print == true
        pretty_table(df, formatters=ft_printf(formatters), title="Confidenc level: $cl", title_alignment = :c, header_crayon = crayon"yellow bold")
    end
    return df
end



"""The Interquartile Range (IQR) is an outlier detection method based on the median and quartiles of the data. IQR represents the range between 
the third quartile (Q3) and the first quartile (Q1), that is: IQR = Q3 — Q1

By setting upper and lower bounds within a certain range, data points can be classified into normal values and outliers:

Lower Bound=Q1−1.5×IQR

Upper Bound=Q3+1.5×IQR

k = 1.5: Mild outliers
k = 3: Extreme outliers

Refernece:

https://medium.com/@tubelwj/python-outlier-detection-iqr-method-and-z-score-implementation-8e825edf4b32
"""
function IQR_outlier_detection(data::Vector{Float64}; k=1.5)
    q1, q3 = Statistics.quantile(data, [0.25, 0.75])
    IQR = q3 - q1
    lower = q1 - k*IQR
    upper = q3 + k*IQR
    return filter(x -> lower <= x <= upper, data)
end

"""
Evaluate the error of data within 1-sigma level
"""
function _STD_(data::AbstractDataFrame; k = 1.5, formatters="%.1f")
    name = names(data)
    std_vec = Float64[]
    for nn in name
        d = IQR_outlier_detection(data[!, nn], k=k)
        push!(std_vec, Statistics.std(d) )
    end
    df = DataFrame(name=name, std=std_vec)
    pretty_table(df, formatters=ft_printf(formatters), title="std err", title_alignment = :c, header_crayon = crayon"yellow bold")
    return df
end


function IQR_outlier_detection(data::AbstractDataFrame; k = 1.5)
    name = names(data)
    data_vec = []
    itr = []
    for nn in name
        d = IQR_outlier_detection(data[!, nn], k=k)
        push!(data_vec, d)
        push!(itr, length(d))
    end
    itr_min = minimum(itr)
    for (i,d) in enumerate(data_vec)
        data_vec[i] = d[[1:itr_min...]]
    end
    return DataFrame(hcat(data_vec...), name)
end