# The calculation of the confidence interval
The function `confidence_interval` is used for calculating the confidence interval of a paramter fixed by fitting within $68\%$, $95\%$ confidence levels.

```julia
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
```

```julia
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
```

```julia
function IQR_outlier_detection(data::Vector{Float64}; k=1.5)
    q1, q3 = Statistics.quantile(data, [0.25, 0.75])
    IQR = q3 - q1
    lower = q1 - k*IQR
    upper = q3 + k*IQR
    return filter(x -> lower <= x <= upper, data)
end
```
```julia
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
```
