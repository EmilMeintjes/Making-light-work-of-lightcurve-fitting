# Light-work-on-lightcurve-fitting
This is a user-friendly Python pipeline that can be used to fit some astronomical light curves.

The idea is that you, as the user, retain the most control over the fitting procedure to lightcurve without the need to write the code yourself. 

The pipeline is laid out as follows:

lightcurve_fitter/
├── models.py       — fit functions + prior builder (no dependencies beyond numpy)

├── persistence.py  — JSON region store + .npy MCMC results

├── selector.py     — Stage 1 interactive region picker

├── initialiser.py  — Stage 2 slider-based parameter initialisation

├── fitter.py       — Stage 3 PyAutoFit/emcee wrapper

├── plots.py        — Stage 4 overview, fit, and corner plots

└── main.py         — CLI entry point wiring all stages together

The pipeline allows you to manually select the exact region which you want to fit some function to. It saves these regions into a JSON file for future reference. Here you can specify the type of function you want to fit (currently available functions are: Gaussian, Rising exponential, Decaying exponential, and Crystal Ball function). 

Once the regions are selected, the user will be prompted to interactively produce an initial guess for the parameters of the chosen function. This is done by adjusting sliders with a min, max and step size similar to what you might see in Desmos. 

You are then prompted to run SciPy's curve_fit (the Levenberg-Marquardt non-linear least squares fitting algorithm) to produce tight initial guesses for the parameters. You can then choose to use your own initial guess or the results from curve_fit as the main guess for the next step. 

A robust MCMC fit using PyAutofit is then performed using a uniform prior in the interval of [0.7 * initial_guess, 1.3 * initial_guess]. The algorithm employs 60 walkers that will complete 1500 steps to search for the best-fitting parameters. The best-fitting parameters are extracted and their uncertainties are estiamted usign the the 16 and 84 percentiles of the posterior distribution, corresponding to a 68% confidence interval.

The pipeline will then produce plot showing the fits to the regions selected and the corner plots from the MCMC searches. 


