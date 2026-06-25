# Hybrid Factor Graph Examples

Examples and sample applications using GTSAM's hybrid factor graphs

## City10000

## Pose Graph Optimization

Compile the code as normal.
You can then run the scripts from the `build/src` directory as

```shell
./hfg_pgo_mc_2d ../../data/g2o/intel.g2o [outlier_percentage] [num_of_trials]
```

So an example would be

```shell
./hfg_pgo_mc_2d ../../data/g2o/intel.g2o 50 10
```

For the DCSAM, GNC and Levenberg-Marquadt baselines, use

```shell
./robust_pgo_mc_2d ../../data/g2o/intel.g2o [outlier_percentage] [num_of_trials]
```

for the Intel (upto 70% outliers) and CSAIL (upto 50% outliers) datasets since they are 2D.

## Plot Comparative Results

```shell
python python/City10000/plot_results.py ISAM2_GT_city10000.txt \
    --estimates results/ISAM2_city10000.txt \
        Hybrid_City10000.txt
```

## Plot Multiple Estimates At Once

```shell
python python/City10000/plot_multiple_estimates.py ISAM2_GT_city10000.txt --estimates \
    multiple_estimates/Hybrid_City10000_100.txt multiple_estimates/Hybrid_City10000_1000.txt \
    multiple_estimates/Hybrid_City10000_2000.txt multiple_estimates/Hybrid_City10000_5000.txt \
    multiple_estimates/Hybrid_City10000_20687.txt --indices 100 1000 2000 5000 10000
```
