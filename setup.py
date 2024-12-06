import setuptools

VERSION = "0.0.1"

setuptools.setup(
    name = "BiGER",
    version = VERSION,
    description = "A Python implementation of BiGER for Bayesian Aggregation in Genomics with Extended Ranking Schemes",
    long_description_content_type = "text/markdown",
    long_description = open("README.md").read(),
    packages=["BiGER"],
    python_requires=">=3.9",
    install_requires=[],
    test_requires=["pytest",
                   "pytest-cov",
                   "pytest-mock",
                   "coverage"],
    classifiers = [
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English"
    ]
)