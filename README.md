# ALTERAPI

# Overview
The performance of data analytics programs has become one of the developers’ major concerns nowadays. We study how API choices could improve data analytics performance while preserving functional equivalence.Based on the natural language processing technology, we have discovered 49 pairs of replaceable API pairs on stackoverflow.We have developed a tool`alterapi` that can help developers discover low-efficiency APIs in the code and recommend higher-efficiency APIs.

`alterapi` can identify low-efficiency APIs from your code.

`alterapi` can recommend higher-efficiency APIs.


# Installing
To install the latest version 

download`alterapi-0.0.2.tar.gz` and then `pip install`

`$ pip install alterapi-0.0.2.tar.gz`



# Example
Here is an example `test.py` using `alterapi`
```python
alterapi has two mode.
'static mode'  just recommend api but  doesn't give execution time.
'dynamic'mode' not only recommend api but also give execution time.
  
import alterapi
print("static mode")
x = alterapi.APIReplace('code.py'，option= 'static' ) 
x.recommend()

print("dynamic mode")
x = alterapi.APIReplace('code.py'，option= 'dynamic' )
x.recommend()
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

original time: 7.540000000005875e-07 Recommend time: 1.1679999999980595e-06

lineno:47


.......
original API:np.sum(t, axis=0)

Recommend API:(t).sum( axis=0)

executing:code_2.py

original time: 0.00019147699999999546, Recommend time:0.00019597100000000366

Recommend API:np.einsum('ij->j',t)

executing:code_2.py

original time: 0.0001643390000000089, Recommend time: 0.00024376200000000736

lineno:74
.......
```
