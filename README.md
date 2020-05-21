# ALTERAPI

# Overview
This tool aims to improve the runtime performance of data manipulation programs.
In particular, it identifies low-efficiency API usages in your source code and 
recommends more efficient alternatives. The current version focuses on APIs from
the `pandas`, `numpy` and `scipy` libraries.


# Installing
Clone this repo and `cd` to its directory, then run
`$ python setup.py install`



# Usage
```python
  
from alterapi import alterapi

# alterapi has two mode.
# 'static' mode (default) analyzes your input code statically without executing it.
tool = alterapi.APIReplace('tests/input.py', option='static') 
tool.recommend()

# 'dynamic' mode makes recommendations by executing your code. This mode also reports the speedup of executing the alternatives.
tool = alterapi.APIReplace('tests/input.py', option='dynamic')
tool.recommend()
```
Results
```
dynamic mode
............
Code at line 10 : df.where(df <= 50, 0)
Recommended code: np.where(df.values <= 50, df.values, 0)
original time:6.5e-04s, new time:1.6e-05s, speedup:40.3x
-------------------------------------------------------------
Code at line 39 : (arr > 30).sum()
Recommended code: np.count_nonzero(arr > 30)
original time:2.2e-05s, new time:8.0e-06s, speedup:2.8x
----------------------------------------------------------------------------
Code at line 55 : np.array(range(10000))
Recommended code: np.arange(10000)
original time:1.0e-03s, new time:4.9e-06s, speedup:210.8x
```
