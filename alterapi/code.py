import numpy as np
import pandas as pd
# test1
a, b = np.random.rand(10, 2), np.random.rand(10, 2)
r1 = np.concatenate([a, b], axis=1)
# should replace with  np.hstack([a,b])

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
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = np.where(df.a > 0, 1, df.a)
# should replace with df.loc[df.a > 0, 'a'] = 1

# test5
(M, K) = (10, 10)
C = np.random.rand(K, K)
X = np.random.rand(M, K)
r1 = np.einsum('ik,km->im', X, C)
# should replace with  np.dot(X, C)


# test6
a = [x for x in range(10)]
r1 = np.fromiter(a, dtype=int)
# should replace with np.array(a)

# test7
A = np.random.rand(1, 10)
r1 = (A > 0.5).astype(int)
# should replace with np.where(A > 0.5, 1, 0)

# test8
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = np.where(df.a > 0, 1, -1)
# should replace with df.apply(lambda row: 1 if row['a'] > 0 else -1, axis=1)

# test9
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df.loc[:, 'a']
# should replace with df.ix[:, 'a']

# test10
df = pd.DataFrame(data=[[list(x)] for x in np.random.rand(10, 2)], columns=['a'])
r1 = df['a'].map(lambda x: x[0])
# should replace with df['a'].apply(lambda x: x[0])

# test11
df = pd.DataFrame(np.random.randn(10, 1))
l = np.random.randint(10, size=50)
r1 = df.iloc[l]
# should replace with  df.loc[l]

# test12
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10)
r1 = df.iat[r, 0]
# should replace with df.iloc[r, 0]

# test13
data = list(map(str, np.random.randint(10, size=10)))
df = pd.DataFrame({'a': data})
r1 = df['a'].astype(np.int64)
# should replace with df['a'].apply(lambda x: int(x))

# test14
dates = pd.date_range('2015', freq='min', periods=10)
dates = [date.strftime('%d %b %Y %H:%M:%S') for date in dates]
r1 = pd.to_datetime(dates)
# should replace with ser.apply(lambda x: parser.parse(x))

# test15
a = np.random.randn(10,1)
r1 = np.sqrt(((a[1:2] - a) ** 2).sum(1))
# should replace with np.linalg.norm(a[1:2] - a, axis=1)

# test16
a = np.random.randn(10, 1)
r1 = np.empty(10);  r1[:] = 0
# should replace with np.full(d, 0)

# test17
a = np.random.randn(10)
r1 = np.count_nonzero(a > 0)
# should replace with np.sum(a > 0)

# test18
df = pd.DataFrame({'A': np.random.randint(1,100,10)})
mapvalue = {i: i+1 for i in range(100)}
r1 = df['A'].map(mapvalue)
# should replace with df['A'].replace(mapvalue)

# test19
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df.loc[df['a'] > 0]
# should replace with  df.query("a>0")

# test20
a = np.random.randn(10)
r1 = np.argmax(a>0)
# should replace with np.where(a>0)[0][0]

# test21
a = np.random.randn(10)
r1 = np.where(a > 0.5)
# should replace with np.nonzero(a > 0.5)

# test22
a = np.random.randint(100, size=(5, 2))
b = np.random.randint(100, size=(5, 2))
r1 = b.dot(a.T)
# should replace with np.tensordot(b, a, axes=((1, 1)))

# test23
indices = np.random.randint(10, size=5)
df1 = pd.DataFrame(np.random.randn(2, 1))
df2 = df1.copy()
df2.iloc[indices, 0] = np.nan
r1 = df1.fillna(df2)
# should replace with df1.combine_first(df2)

# test24
A = np.arange(10)
r1 = np.append(A, A)
# should replace with np.hstack((A,A))

# test25
A = np.random.rand(10, 1)
B = np.random.rand(10, 1)
r1 = np.hstack((A, B))
# should  replace with np.c_[A, B]

# test26
r1 = np.zeros(10)
# should replace with np.full(10, 0)

# test27
a = np.random.normal(size=(1, 10))
r1 = np.concatenate((a, a), axis=0)
# should replace with np.vstack((a, a)

# test28
x = np.random.randint(10, size=10).reshape(5, 2)
y = np.arange(5)
r1 = x * y.reshape(-1, 1)
# should replace with  x * np.atleast_2d(y).T

# test29
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r = np.random.randint(0, 10)
r1 = df['a'].iat[r]
# should replace with df['a'].iloc[r]

# test30
arr = np.random.randn(10, 1)
df = pd.DataFrame(arr)
r1 = np.cumprod(1 + arr, axis=0) - 1
# should replace with((1 + df).cumprod(axis=0) - 1)

# test31
df = pd.DataFrame(columns=['id', 'category'])
df['id'] = np.random.randint(3, size=10)
df['category'] = np.random.choice(['a','b','c'], )
r1 = df.pivot_table(index='id', columns='category', aggfunc=len, fill_value=0)
# should replace with pd.crosstab(df['id'], df['category'])

# test32
data = list(map(str, np.random.randint(10, size=10)))
df = pd.DataFrame({'a': data})
r1 = df['a'].map(lambda x: x[0])
# should replace with df["a"].str[0]

# test33
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = np.where(df.a > 0, 1, -1)
# should replace with  df.a.map(lambda x: 1 if x > 0 else -1)

# test34
a = np.arange(10).reshape((-1, 5))
r1 = np.sqrt(np.einsum('ij,ij->i', a, a))
# should replace with np.sqrt((a*a).sum(axis=1))

# test35
a = np.random.randint(10, size=10)
r1 = np.transpose([a,a])
# should replace with np.column_stack((a,a))

# test36
r1 = np.empty(10); r1[:] = 0
# should replace with np.zeros(10)

# test37
s = pd.Series(np.random.randint(1,100,10))
r1 = pd.Series([[a] for a in s])
# should replace with s.apply(lambda x: [x])

# test38
a = np.random.randint(10, size=10)
r1 = np.vstack((a, a)).T
# should replace with np.column_stack((a,a))

# test39
r1 = np.empty(10); r1[:] = 1
# should replace with np.ones(10)

# test40
df = pd.DataFrame(np.random.randn(10, 1), columns=['a'])
r1 = df["a"].astype(str)
# should replace with df["a"].map(str)

# test41
a = np.random.randint(10, size=10)
r1 = np.array(a*2)
# should replace with np.hstack(a*2)

# test42
arr = np.arange(10)
r1 = np.repeat(arr[None,:], 2, axis=0)
# should replace with np.tile(arr, (2, 1, 1))

# test43
arr = np.arange(10)
r1 = arr.sum()
# should replace with np.sum(arr)

# test44
df = pd.DataFrame(dict(gender=[f"mostly_{g}" for g in ['male', 'female'] * 10]))
r1 = df.gender.map({'mostly_male': 'male', 'mostly_female': 'female'})
# should replace with df.replace({'gender': {'mostly_': ''}}, regex=True)

# test45
df = pd.DataFrame(np.arange(0, 3, 1, ), columns=["a"])
r1 = df.apply(lambda x: x * 2)
# should replace with df.a.apply(lambda x: x * 2)

# test46
df = pd.DataFrame(np.random.randint(100, size=(10,3)))
r1 = np.where(df.values <= 50, df.values, np.nan)
# should replace with df.where(df <= 50)

# test47
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10)
r1 = df.iat[r, 0]
# should replace with df.loc[r, 0]

# test48
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10)
r1 = df.at[r, 0]
# should replace with df.iloc[r, 0]

# test49
df = pd.DataFrame(np.random.randn(10, 1))
r = np.random.randint(0, 10 - 1)
r1 = df.at[r, 0]
# should replace with  df.iat[r, 0]

# test50
item_ids = [1, 2, 3, 4]
df = pd.DataFrame({'item_id': np.random.choice(item_ids + [5], p=(.125, .125, .125, .125, .5), size=10)})
r1 = df[df['item_id'].isin(item_ids)]
# should replace with df.query('item_id in {}'.format(item_ids))






























