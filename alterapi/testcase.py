import numpy as np
import pandas as pd
# test1
a, b = np.random.rand(10, 2), np.random.rand(10, 2)
r1 = np.hstack([a,b])
# should replace with np.concatenate([a, b], axis=1)

# test2
a = np.random.random_sample(100)
r1 = np.count_nonzero(a > 0.5)
# should replace with  np.sum(a>0.5)

# test3
a = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r = np.random.randint(0, 10)
r1 = a.at[r, 'a']
# should replace with a.loc[r,'a']


# test4
a = [x for x in range(10)]
r1 = np.fromiter(a, dtype=int)
# should replace with np.array(a, dtype=int)

# test5
a = [x for x in range(10)]
r1 = np.array(a, dtype=int)
# should replace with np.fromiter(a, dtype=int)


# test6
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = np.where(df.a > 0, 1, -1)
# should replace with df.apply(lambda row: 1 if row['a'] > 0 else -1, axis=1)
# should replace with df.a.map(lambda x: 1 if x > 0 else -1)

# test7
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df.loc[:, 'a']
# should replace with df.ix[:, 'a']

# test8
df = pd.DataFrame(data=[[list(x)] for x in np.random.rand(10, 2)], columns=['a'])
r1 = df['a'].apply(lambda x: x[0])
# should replace with  df['a'].map(lambda x: x[0])

# test9
df = pd.DataFrame(np.random.randn(10, 1))
l = np.random.randint(10, size=50)
r1 = df.iloc[l]
# should replace with  df.loc[l]

# test10
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10)
r1 = df.iat[r, 0]
# should replace with df.iloc[r, 0]

# test11
data = list(map(str, np.random.randint(10, size=10)))
df = pd.DataFrame({'a': data})
r1 = df['a'].astype(np.int64)
# should replace with df['a'].apply(lambda x: int(x))



# test12
a = np.random.randn(10, 1)
r1 = np.full((10, 2), 0, dtype=np.int16)
# should replace with np.empty((10,2), dtype=np.int16);  r1[:] = 0

# test13
df = pd.DataFrame({'A': np.random.randint(1,100,10)})
mapvalue = {i: i+1 for i in range(100)}
r1 = df['A'].map(mapvalue)
# should replace with df['A'].replace(mapvalue)

# test14
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df.loc[df['a'] > 0]
# should replace with  df.query("a>0")


# test15
a = np.random.randn(10)
r1 = np.where(a > 0.5)
# should replace with np.nonzero(a > 0.5)

# test16
a = np.random.randint(100, size=(5, 2))
b = np.random.randint(100, size=(5, 2))
r1 = b.dot(a.T)
# should replace with np.tensordot(b,a.T,axes=1))

# test17
indices = np.random.randint(2, size=5)
df1 = pd.DataFrame(np.random.randn(2, 1))
df2 = df1.copy()
df2.iloc[indices, 0] = np.nan
r1 = df1.fillna(df2)
# should replace with df1.combine_first(df2)


# test18
A = np.arange(10)
r1 = np.hstack((A, A))
# should replace with np.append(A, A)

# test19
A = np.random.rand(10, 1)
B = np.random.rand(10, 1)
r1 = np.c_[A, B]
# should  replace with  np.hstack((A, B))

# test20
r1 = np.full((10,2), 0)
# should replace with np.zeros((10,2))

# test21
a = np.random.normal(size=(1, 10))
r1 = np.vstack((a, a))
# should replace with np.concatenate((a, a), axis=0)

# test22
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r = np.random.randint(0, 10)
r1 = df['a'].iat[r]
# should replace with df['a'].iloc[r]


# test23
data = list(map(str, np.random.randint(10, size=10)))
df = pd.DataFrame({'a': data})
r1 = df["a"].str[0]
# should replace with df['a'].map(lambda x: x[0])

# test24
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = np.where(df.a > 0, 1, -1)
# should replace with  df.a.map(lambda x: 1 if x > 0 else -1)


# test25
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df.a.map(lambda x: 1 if x > 0 else -1)
# should replace with  np.where(df.a > 0, 1, -1)

# test26
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = np.where(df.a > 0, 1, -1)
# should replace with df.a.map(lambda x: 1 if x > 0 else -1)

# test27
a = np.random.randint(10, size=10)
r1 = np.transpose([a,a])
# should replace with np.column_stack((a,a))

# test28
r1 = np.zeros((10, 2))
# should replace with np.empty((10,2)); r1[:] = 0


# test29
a = np.random.randint(10, size=10)
r1 = np.vstack((a, a))
# should replace with np.column_stack((a,a)).T

# test30
a = np.random.randint(10, size=10)
r1 = np.column_stack((a,a))
# should replace with np.vstack((a, a)).T

# test31
r1 = np.ones((10,1))
# should replace with np.empty(10); r1[:] = 1

# test32
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df["a"].astype(str)
# should replace with df["a"].map(str)

# test33
a = np.random.randint(10, size=10)
r1 = np.array(a*2)
# should replace with np.hstack(a*2)


# test34
arr = np.arange(10)
r1 = arr.sum()
# should replace with np.sum(arr)

# test35
df = pd.DataFrame(dict(gender=[f"mostly_{g}" for g in ['male', 'female'] * 10]))
r1 = df.gender.map({'mostly_male': 'male', 'mostly_female': 'female'})
# should replace with df.gender.replace({'mostly_male': 'male', 'mostly_female': 'female'})


# test36
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10)
r1 = df.iat[r, 0]
# should replace with df.loc[r, 0]

# test37
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10)
r1 = df.at[r, 0]
# should replace with df.iloc[r, 0]

# test38
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10 - 1)
r1 = df.at[r, 0]
# should replace with  df.iat[r, 0]

# test39
a = np.arange(10*2).reshape(10,2)
r1 = np.sum(a)
# should replace with np.einsum('ij->', a)




























































