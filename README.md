# Where to measure next

Interactive pages for three papers on boundary bias in Gaussian processes:

- Boundary Variance Inflation Causes Acquisition Bias in Gaussian Processes ([paper](https://arxiv.org/abs/2606.07561), [code](https://github.com/mariabankestad/gp-boundary-bias))
- Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design ([paper](https://arxiv.org/abs/2602.01898), [code](https://github.com/DiffUse-AI/warp_gp))
- Understanding Hidden Spatial Preferences in Bayesian Optimization ([paper](https://arxiv.org/abs/2610.14696), [code](https://github.com/mariabankestad/spatial-bo))

The site is the `docs/` folder: plain HTML, CSS and JavaScript, no build step.

    open docs/index.html

The files in `docs/data/` are written by the scripts in `precompute/` from the released code of the papers:

    python precompute/pack_data.py <warp_gp dir> <spatial-bo dir>
    python precompute/pack_density.py <gp-boundary-bias dir>
