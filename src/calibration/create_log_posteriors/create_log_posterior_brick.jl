using Missings
using DataFrames
using Distributions
using NetCDF
using Turing
using PDMats
using Bijectors
using Random
using LinearAlgebra

#-------------------------------------------------------------------------------
# This file contains functions used to calculate the log-posterior for the BRICK model.
#-------------------------------------------------------------------------------

"""
    construct_brick_log_prior(joint_antarctic_prior::Bool; calibration_data_dir::Union{String, Nothing} = nothing)

Calculate total (log) prior probability for brick.

Description: This creates a function that will calculate the total (log) prior probability of the uncertain model,
             initial condition, and statistical process parameters specific to the standalone BRICK model. It uses
             non-uniform priors for the Antarctic ice sheet parameters, informed by a previous model calibration to
             paleo data. There are two options for the Antarctic priors (1) fitting marginal distributions using a
             kernel density estimation or (2) fitting a multivariate normal distribution that accounts for correlations
             that emerge during the paleo calibration (note, many of the marginal paleo pdfs are not normally distributed).

Function Arguments:

      joint_antarctic_prior   = TRUE/FALSE check for whether to use a joint normal prior distribution (TRUE = option 1 described
                              above) or fitted marginal kernel density estimates (FLASE = option 2 described above).
      calibration_data_dir    = Data directory for calibration data. Defaults to package calibration data directory, 
                                changing this is not recommended.
"""

# antarctic prior
struct AntarcticPrior <: ContinuousMultivariateDistribution end

function construct_antarctic_prior(; calibration_data_dir::Union{String, Nothing} = nothing)

    # set calibration data directory if one was not provided ie. it is set as nothing
    if isnothing(calibration_data_dir)
        calibration_data_dir = joinpath(@__DIR__, "..", "..", "..", "data", "calibration_data")
    end

    # Load required data to create Antarctic ice sheet informative priors (posterior parameters from previous calibration to paleo data).
    # Note: this excludes the Antarctic variance term because the model uses an AR(1) model for the recent instrumental observations.
    #       From original BRICK Fortran/R code: "var.dais was fit to paleo data-model mismatch, not representative of the current era."
    antarctic_paleo_file   = joinpath(calibration_data_dir, "DAISfastdyn_calibratedParameters_gamma_29Jan2017.nc")
    antarctic_paleo_params = convert(Array{Float64,2}, ncread(antarctic_paleo_file, "DAIS_parameters"))'[:,1:15]

    #---------------------------------------------
    # Antarctic ice sheet priors
    #---------------------------------------------

    # Calculate upper and lower bounds for Antarctic ice sheet parameters (min/max values from paleo calibration).
    antarctic_lower_bound = vec(minimum(antarctic_paleo_params, dims=1))
    antarctic_upper_bound = vec(maximum(antarctic_paleo_params, dims=1))

    # Fit a multivariate normal to the data (do not use variance estimate for paleo data, separately estimate AR(1) parameters for recent observations isntead).
    antarctic_joint_prior = fit(MvNormal, antarctic_paleo_params')
    
    function rand_func!(rng::AbstractRNG, d::AntarcticPrior, x::AbstractVector{<:Real})
        rand!(rng, antarctic_joint_prior, x)
        while any(x .< antarctic_lower_bound) || any(x .> antarctic_upper_bound)
            rand!(rng, antarctic_joint_prior, x)
        end               
        return x
    end

    function log_likelihood_func(d::AntarcticPrior, x::AbstractVector) 
        if any(x .< antarctic_lower_bound) || any(x .> antarctic_upper_bound)
            return -Inf
        else
            return logpdf(antarctic_joint_prior, x)
        end
    end

    bijector_func(d::AntarcticPrior) = Stacked(Bijectors.Logit.(antarctic_lower_bound, antarctic_upper_bound))

    return (rand_func!, log_likelihood_func, bijector_func)        
end

antarctic_dist_funcs = construct_antarctic_prior()
Base.length(d::AntarcticPrior) = 15
Distributions._rand!(rng::AbstractRNG, d::AntarcticPrior, x::AbstractArray{<:Real}) = antarctic_dist_funcs[1](rng, d, x)
Distributions.logpdf(d::AntarcticPrior, x::AbstractArray{<:Real}) = antarctic_dist_funcs[2](d, x)
Bijectors.bijector(d::AntarcticPrior) = antarctic_dist_funcs[3]

function get_brick_calibration_data(model_start_year::Int=1850, calibration_end_year::Int=2017)

# Create a vector of calibration years and calculate total number of years to run model.
    calibration_years = collect(model_start_year:calibration_end_year)
    n = length(calibration_years)

    # Load calibration data/observations.
    calibration_data, obs_antarctic_trends, obs_thermal_trends = MimiBRICK.load_calibration_data(model_start_year, calibration_end_year, last_sea_level_norm_year=1990)

    return calibration_data, obs_antarctic_trends, obs_thermal_trends
end

calibration_data, antarctic_trends, thermal_trends = get_brick_calibration_data()

function get_calibration_inputs(calibration_data::DataFrame, thermal_trends::DataFrame)
    ## get calibration data indices 
    # Calculate indices for each year that has an observation in calibration data sets.
    indices_glaciers_data      = findall(x-> !ismissing(x), calibration_data.glaciers_obs)
    n_glaciers = length(indices_glaciers_data)
    indices_greenland_data     = findall(x-> !ismissing(x), calibration_data.merged_greenland_obs) # Use merged Greenland data.
    n_greenland = length(indices_greenland_data)
    indices_antarctic_data     = findall(x-> !ismissing(x), calibration_data.antarctic_imbie_obs)
    n_antarctic = length(indices_antarctic_data)
    indices_gmsl_data          = findall(x-> !ismissing(x), calibration_data.gmsl_obs)
    n_gmsl = length(indices_gmsl_data)
    n_thermal_trend = size(thermal_trends, 1)
    obs_lengths = disallowmissing([n_glaciers, n_greenland, n_antarctic, n_gmsl, n_thermal_trend])

    ## get observations
    obs_antarctica = calibration_data[indices_antarctic_data, :antarctic_imbie_obs]
    obs_greenland = calibration_data[indices_greenland_data, :merged_greenland_obs]
    obs_glaciers = calibration_data[indices_glaciers_data, :glaciers_obs]
    obs_gmsl = calibration_data[indices_gmsl_data, :gmsl_obs]
    obs_thermal_trend = thermal_trends.Trend
    obs = disallowmissing([obs_glaciers; obs_greenland; obs_antarctica; obs_gmsl; obs_thermal_trend])

    ## get observation errors
    err_glaciers = calibration_data[indices_glaciers_data, :glaciers_sigma]
    err_greenland = calibration_data[indices_greenland_data, :merged_greenland_sigma]
    err_antarctic = calibration_data[indices_antarctic_data, :antarctic_imbie_sigma]
    err_gmsl = calibration_data[indices_gmsl_data, :gmsl_sigma]
    # Calculate σ for observed trends based on IPCC 90% trend window values.
    err_thermal_trend = 0.5 .* (thermal_trends.Upper_90_Percent .- thermal_trends.Lower_90_Percent)
    err = Diagonal(disallowmissing([err_glaciers; err_greenland; err_antarctic; err_gmsl; err_thermal_trend]))

    return (obs, err, obs_lengths)
end

@model function brick_posterior(observations, obs_error, obs_lengths, thermal_trends, f_run_model; model_start_year::Int=1850, calibration_end_year::Int=2017)
    ## priors
    σ_glaciers ~ Uniform(1e-10, 0.0015) # Based on BRICK code.
    σ_greenland ~ Uniform(1e-10, 0.002) # Based on BRICK code.
    σ_antarctic ~ Uniform(1e-10, 0.063) # Based on BRICK code.
    σ_gmsl ~ Uniform(1e-10, 0.05) # Just setting the same as prior_σ_gmsl_1900 value from old BRICK code.
    ρ_glaciers ~ Uniform(-0.99, 0.99)
    ρ_greenland ~ Uniform(-0.99, 0.99)
    ρ_antarctic ~ Uniform(-0.99, 0.99)
    ρ_gmsl ~ truncated(Normal(0.8, .25), -1.0, 1.0)

    thermal_s₀ ~ Uniform(-0.0484, 0.0484) # BRICK defaults. # Initial sea level rise due to thermal expansion designated in 1850 (m SLE).
    greenland_v₀ ~ Uniform(7.16, 7.56)
    glaciers_v₀ ~ Uniform(0.31, 0.53)
    glaciers_s₀ ~ Uniform(-0.0536, 0.0791)
    antarctic_s₀ ~ Uniform(-0.04755, 0.05585) # Informed by prior BRICK runs.

    thermal_α ~ Uniform(0.05, 0.3) # upper/lower bounds from "Impacts of Observational Constraints Related to Sea Level on Estimates of Climate Sensitivity"  # Global ocean-averaged thermal expansion coefficient (kg m⁻³ °C⁻¹).

    greenland_a ~ Uniform(-4.0, -0.001)
    greenland_b ~ Uniform(5.888, 8.832)
    greenland_α ~ Uniform(0.0, 0.001)
    greenland_β ~ Uniform(0.0, 0.001)
    glaciers_β₀ ~ Uniform(0.0, 0.041)
    glaciers_n  ~ Uniform(0.55, 1.0)
    antarctic_params ~ AntarcticPrior()

    (anto_α,
    anto_β,
    antarctic_γ,
    antarctic_α,
    antarctic_μ,
    antarctic_ν,
    antarctic_precip₀,
    antarctic_κ,
    antarctic_flow₀,
    antarctic_runoff_height₀,
    antarctic_c,
    antarctic_bedheight₀,
    antarctic_slope,
    antarctic_λ,
    antarctic_temp_threshold) = antarctic_params

    # run model    
    (modeled_glaciers, 
    modeled_greenland, 
    modeled_antarctic, 
    modeled_thermal_expansion, 
    modeled_gmsl) = f_run_model(
                        (thermal_s₀,
                        greenland_v₀,
                        glaciers_v₀,
                        glaciers_s₀,
                        antarctic_s₀,
                        thermal_α,
                        greenland_a,
                        greenland_b,
                        greenland_α,
                        greenland_β,
                        glaciers_β₀,
                        glaciers_n,
                        anto_α,
                        anto_β,
                        antarctic_γ,
                        antarctic_α,
                        antarctic_μ,
                        antarctic_ν,
                        antarctic_precip₀,
                        antarctic_κ,
                        antarctic_flow₀,
                        antarctic_runoff_height₀,
                        antarctic_c,
                        antarctic_bedheight₀,
                        antarctic_slope,
                        antarctic_λ,
                        antarctic_temp_threshold)
                    )

    break_indices = cumsum(obs_lengths)
    n_all = sum(obs_lengths)             

    # Calculate the AIS trends (in milimeters) from the annual modeled output.
    modeled_thermal_trend = calculate_trends(disallowmissing(modeled_thermal_expansion), thermal_trends, model_start_year, calibration_end_year)

    ## construct covariance matrices for each observation series
    H_glaciers = abs.((1:obs_lengths[1]) .- (1:obs_lengths[1])')
    Σ_glaciers = ((σ_glaciers / sqrt(1 - ρ_glaciers^2)) .^ H_glaciers)
    H_greenland = abs.((1:obs_lengths[2]) .- (1:obs_lengths[2])')
    Σ_greenland = ((σ_greenland / sqrt(1 - ρ_greenland^2)) .^ H_greenland) 
    H_antarctic = abs.((1:obs_lengths[3]) .- (1:obs_lengths[3])')
    Σ_antarctic = ((σ_antarctic / sqrt(1 - ρ_antarctic^2)) .^ H_antarctic)
    H_gmsl = abs.((1:obs_lengths[4]) .- (1:obs_lengths[4])')
    Σ_gmsl = ((σ_gmsl / sqrt(1 - ρ_gmsl^2)) .^ H_gmsl)

    # combine time series and compute joint likelihood
    Σ = zeros(n_all, n_all)
    Σ[1:break_indices[1], 1:break_indices[1]] = Σ_glaciers
    Σ[break_indices[1]+1:break_indices[2], break_indices[1]+1:break_indices[2]] = Σ_greenland
    Σ[break_indices[2]+1:break_indices[3], break_indices[2]+1:break_indices[3]] = Σ_antarctic
    Σ[break_indices[3]+1:break_indices[4], break_indices[3]+1:break_indices[4]] = Σ_gmsl
    for i = 1:obs_lengths[5]
        Σ[break_indices[4]+i, break_indices[4]+i] = obs[break_indices[4]+i]
    end
    Σ += obs_error

    modeled_all = [modeled_glaciers; modeled_greenland; modeled_antarctic; modeled_gmsl; modeled_thermal_trend]
    modeled_all ~ MvNormal(observations, PDMats.Symmetric(Σ))


end

run_brick = construct_run_brick(1850, 2017)
(obs, err, obs_length) = get_calibration_inputs(calibration_data, thermal_trends)
model = brick_posterior(obs, err, obs_length, thermal_trends, run_brick)
chain = sample(model, NUTS(), 100)

##------------------------------------------------------------------------------
## End
##------------------------------------------------------------------------------
