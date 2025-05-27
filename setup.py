from setuptools import setup, find_packages


setup(
        name='LDD2025',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/LDD2025',
        description="Experiments for LDD 2025",
        packages=find_packages(include=[
                'dexterous_bioprosthesis_2021_raw_datasets_framework',
                'dexterous_bioprosthesis_2021_raw_datasets_framework.*',
                'dexterous_bioprosthesis_2021_raw_datasets_framework_experiments'
                ]),
        install_requires=[ 
                'pandas>=2.2.2',
                'numpy==1.23.5',
                'matplotlib==3.9.4',
                'scipy==1.12.0',
                'joblib==1.4.2',
                'scikit-learn==1.2.2',
                'tqdm==4.66.4',
                'joblib==1.4.2', 
                'DESlib==0.3.5',
                'numba==0.60.0',
                'statsmodels==0.13.5',
                'PyWavelets==1.4.1',
                'pt_outlier_probability @ git+https://github.com/ptrajdos/ptOutlierProbability.git@4d137a12220612ed6078178a6cf54b4c98699d99',
                'tabulate==0.9.0',
                'Jinja2==3.1.2',
                'seaborn==0.13.2',
                'weightedkjnn @ git+https://github.com/ptrajdos/WeightedKNNClassifier.git',
                'results-storage @ git+https://github.com/ptrajdos/results_storge.git',
                'kernelnb @ git+https://github.com/ptrajdos/KernelNB.git',
                
        ],
        test_suite='test'
        )
