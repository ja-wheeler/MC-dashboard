from setuptools import setup, find_packages

setup(
    name="mc-dashboard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.2.3,<3.0.0',
        'numpy>=1.26.0,<2.0.0',
        'scipy>=1.11.3,<2.0.0',
        'matplotlib>=3.8.0,<4.0.0',
        'seaborn>=0.12.2,<1.0.0',
        'plotly>=5.15.0,<6.0.0',
        'scikit-learn>=1.3.1,<2.0.0',
        'xgboost>=1.7.6,<2.0.0',
        'tensorflow>=2.14.0,<3.0.0',
        'statsmodels>=0.14.0,<1.0.0',
        'streamlit>=1.41.0,<2.0.0',
        'typing-extensions>=4.12.2,<5.0.0',
    ],
)