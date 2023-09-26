# Locally differentially Private Decision Tree (LPDT)

This repository implements **LPDT**, the Locally differentially Private Decision Tree described in the paper Decision Tree for Locally Private Estimation with Public Data accepted for NeurIPS 2023.
The implementation is based on pure Python with the following required packages:

- Scikit-learn
- NumPy
- Numba
- Scipy

## Contents

- [Installation](#Installation)
- [Demo](#Demo)
- [References](#References)


## Installation

### Via PyPI

```bash
pip install LPDT
```

### Via GitHub

```bash
pip install git+https://github.com/Karlmyh/LPDT.git
```


### Manual Install
  > 
```bash
git clone git@github.com:Karlmyh/LPDT.git
cd LPDT 
python setup.py install
```

## Demo
See simulation.py for a demo of the use of the class.



## References

- *Decision Tree for Locally Private Estimation with Public Data* (NeurIPS 2023)
