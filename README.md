# pdmp_jax


This repository contains a JAX implementation of the PDMP samplers. 

It contains the following PDMP samplers:
- Zig-Zag sampler (Joris Bierkens, Paul Fearnhead, Gareth Roberts. "The Zig-Zag process and super-efficient sampling for Bayesian analysis of big data." The Annals of Statistics, 47(3) 1288-1320 June 2019. https://doi.org/10.1214/18-AOS1715)
- Bouncy Particle Sampler (Bouchard-Côté, A., Vollmer, S. J., & Doucet, A. (2018). The Bouncy Particle Sampler: A Nonreversible Rejection-Free Markov Chain Monte Carlo Method. Journal of the American Statistical Association, 113(522), 855–867. https://doi.org/10.1080/01621459.2017.1294075)
- Forward Event Chain (Forward Ref with random time for the orthogonal switch) (Michel, M., Durmus, A., & Sénécal, S. (2020). Forward Event-Chain Monte Carlo: Fast Sampling by Randomness Control in Irreversible Markov Chains. Journal of Computational and Graphical Statistics, 29(4), 689–702. https://doi.org/10.1080/10618600.2020.1750417)
- Speedup ZigZag (non explosive case) (G. Vasdekis, G. O. Roberts. "Speed up Zig-Zag." The Annals of Applied Probability, 33(6A) 4693-4746 December 2023. https://doi.org/10.1214/23-AAP1930)
- Boomerang Sampler (Bierkens, J., Grazzi, S., Kamatani, K. &amp; Roberts, G.. (2020). The Boomerang Sampler. <i>Proceedings of the 37th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 119:908-918 Available from https://proceedings.mlr.press/v119/bierkens20a.html )


The implementation relies on a general class PDMP that is used to define all the PDMP samplers, making it easy to add new PDMP samplers.

It can be installed using pip:
```bash
pip install pdmp-jax
```

The file `example.ipynb` contains an example of how to use the package.