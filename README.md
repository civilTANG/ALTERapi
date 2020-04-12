# api_replacement

# Overview
The performance of data analytics programs has become one of the developers’ major concerns nowadays. We study how API choices could improve data analytics performance while preserving functional equivalence.Based on the natural language processing technology, we have discovered 49 pairs of replaceable API pairs on stackoverflow.We have developed a tool`api_replacement` that can help developers discover low-efficiency APIs in the code and recommend higher-efficiency APIs.

`api_replacement` can identify low-efficiency APIs from your code.

`api_replacement` can recommend higher-efficiency APIs.


# Installing
To install the latest version from GitHub

`$ pip install git+git://github.com/civilTANG/APIreplacement.git`



# Example
Here is an example `test.py` using `api_replacement`
```python
import api_replacement
x = api_replacement.APIReplace('code.py') # https://www.kaggle.com/aawadall/deep-neural-net
x.find('code.py')
```
result

```
.......

Original API:np.dot(W, filter1)

Recommend API:np.tensordot(np,W, filter1,axes=1)

Recommend API:np.einsum('ij,jm->im',W, filter1)

lineno:46

........
```
