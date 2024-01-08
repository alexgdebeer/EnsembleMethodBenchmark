using HDF5 

enrml_folder = "data/enrml"
eki_folder = "data/eki"

fnames = [

    # "$(enrml_folder)/enrml_100", 
    # "$(enrml_folder)/enrml_boot_100", 
    # "$(enrml_folder)/enrml_shuffle_100", 
    # "$(enrml_folder)/enrml_fisher_100",
    # "$(enrml_folder)/enrml_inflation_100",
    # "$(enrml_folder)/enrml_fisher_inflation_100",

    # "$(eki_folder)/eki_100",
    # "$(eki_folder)/eki_boot_100",
    # "$(eki_folder)/eki_shuffle_100",
    "$(eki_folder)/eki_sec_100",
    "$(eki_folder)/eki_fisher_100",
    "$(eki_folder)/eki_inflation_100",
    "$(eki_folder)/eki_fisher_inflation_100"

]

n_trials = 10

for fname ∈ fnames
    f = h5open("$(fname).h5", "r")
    f_new = h5open("$(fname)_new.h5", "w")
    print(keys(f))
    for i ∈ 1:n_trials

        μ_post = transform(pr, mean(f["θs_$i"][:, :], dims=2))
        μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)
        
        σ_post = std(f["us_$i"][:, :], dims=2)
        σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)

        f_new["θs_$i"] = f["θs_$i"][:, :]
        f_new["us_$i"] = f["us_$i"][:, :]
        f_new["Gs_$i"] = f["Gs_$i"][:, :]

        f_new["μ_post_$i"] = μ_post
        f_new["σ_post_$i"] = σ_post

        f_new["n_its_$i"] = read(f["n_its_$i"])
        f_new["n_sims_$i"] = read(f["n_sims_$i"])

    end
    close(f)
    close(f_new)

end