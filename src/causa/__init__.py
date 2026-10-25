"""causa: causal inference and experimentation for growth teams.

Two questions drive the package. Did last quarter's targeted campaign work, given that marketers chose who
received it? And should the product team ship the variant of a running A/B test? The first needs
propensity models, weighting and doubly robust estimators on confounded data; the second needs variance
reduction, sequential tests that tolerate peeking, and checks that the randomisation itself is healthy.
Synthetic generators with known ground truth (:mod:`causa.data`) make every estimator measurable.
"""

__version__ = "0.1.0"
