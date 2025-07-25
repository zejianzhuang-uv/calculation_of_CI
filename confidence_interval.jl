

import Statistics
import Distributions
using PrettyTables

"""
Refernece:

https://rpubs.com/JuliaWorkshop/Confidence_Intervals_Julia
"""
function confidence_interval(data::Vector{Float64}; k=3, cl=68e-2)
    filter_data = IQR_outlier_detection(data, k=k)
    # mean = Statistics.mean(filter_data)
    std = Statistics.std(filter_data)

    alpha = 1. - cl
    t = Statistics.quantile(Distributions.TDist(length(filter_data) - 1), 1- alpha/2)
    # Calculate margin of error
    merr = t * (std / sqrt(length(filter_data) ) )
    return merr
end

function confidence_interval(data::AbstractDataFrame; k=3, cl=68e-2, formatters="%.3f", pretty_print=true)
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
function IQR_outlier_detection(data::Vector{Float64}; k=3)
    q1, q3 = Statistics.quantile(data, [0.25, 0.75])
    IQR = q3 - q1
    lower = q1 - k*IQR
    upper = q3 + k*IQR
    return filter(x -> lower <= x <= upper, data)
end