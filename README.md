# ALTERAPI

# Overview
The performance of data analytics programs has become one of the developers’ major concerns nowadays. We study how API choices could improve data analytics performance while preserving functional equivalence.Based on the natural language processing technology, we have discovered 49 pairs of replaceable API pairs on stackoverflow.We have developed a tool`alterapi` that can help developers discover low-efficiency APIs in the code and recommend higher-efficiency APIs.

`alterapi` can identify low-efficiency APIs from your code.

`alterapi` can recommend higher-efficiency APIs.


# Installing
To install the latest version 

download`alterapi-0.0.1.tar.gz` and then `pip install`

`$ pip install alterapi-0.0.1.tar.gz`



# Example
Here is an example `test.py` using `alterapi`
```python
import alterapi
print("static mode")
x = alterapi.APIReplace('code.py'，option= 'static' )
x.recommend()

print("dynamic mode")
x = alterapi.APIReplace('code.py'，option= 'dynamic' )
x.recommend())
```
result

```
static mode
.......

original API:np.zeros((n[l], 1))

Recommend API:np.empty((n[l], 1)); r2[:]= 0

lineno:47

.......

Original API:np.dot(W, filter1)

Recommend API:np.tensordot(np,W, filter1,axes=1)

Recommend API:np.einsum('ij,jm->im',W, filter1)

lineno:46

........

original API:df.astype(int)

Recommend API:df.apply(int)

lineno:502

........

```
```
dynamic mode
............
original API:np.zeros((n[l], 1))
Recommend API:np.empty((n[l], 1)); r2[:]= 0
executing:code_1.py
lineno:47
47,"np.zeros((n[l], 1))","np.empty((n[l], 1)); r2[:]= 0",7.540000000005875e-07,1.1679999999980595e-06
.......
```
