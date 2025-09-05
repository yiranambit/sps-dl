import codecs

from setuptools import find_packages
from setuptools import setup


requirements = [
    "numpy >= 1.26.4",
    "pandas >= 2.2.2",
    "tensorflow == 2.16.1",
    "scikit-learn >= 1.4.2",
    "scipy >= 1.13.0",
    "pandera >= 0.19.3",
    "polars >= 0.20.30",
    "altair >= 5.3.0",
    "tqdm >= 4.66.4",
]

extras_require = {
    "demos": ["jupyterlab >= 4.2.0", "pyhpo >= 3.3.0"],
    "scripts": ["hydra-core >= 1.3.2", "hydra-optuna-sweeper >= 1.2.0"],
}

with open("./test-requirements.txt") as test_reqs_txt:
    test_requirements = [line for line in test_reqs_txt]


long_description = ""
with codecs.open("./README.md", encoding="utf-8") as readme_md:
    long_description = readme_md.read()

setup(
    name="lpm",
    use_scm_version={"write_to": "lpm/_version.py"},
    description="Ambit longitudinal phenotype modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csederman/ambit-lpm",
    packages=find_packages(exclude=["tests.*", "tests"]),
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require=extras_require,
    python_requires=">=3.11",
    zip_safe=False,
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
    ],
    maintainer="Casey Sederman",
    maintainer_email="casey.sederman@hsc.utah.edu",
)
