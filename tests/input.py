import numpy as np
import pandas as pd
import string

df = pd.DataFrame(np.random.randint(0,100,size=(10,10)))
r1 = df.loc[5, 8]
r2 = df.iloc[5, 8]
r3 = df.loc[np.arange(5)]
r4 = df.where(df <= 50)
r5 = df.where(df <= 50, 0)
r6 = df.cumprod(axis=1)
r7 = df.cumprod()

df = pd.DataFrame(np.random.randint(0, 100, size=(26, 26)), columns=list(string.ascii_lowercase))
r8 = df.loc[5, 'a']
r9 = df['a'].map(lambda x: 1 if x > 50 else -1)
r10 = df.apply(lambda row: 1 if row['a'] > 50 else -1, axis=1)


A, B = np.random.rand(10, 50), np.random.rand(10, 50)
r11 = np.c_[A, B]


df = pd.DataFrame([[np.random.choice(range(5)) for x in range(20)] for y in range(30)])
r12 = df[5].replace({i: i + 1 for i in range(5)})
r13 = df[3].apply(lambda x: x + 2)
r14 = df[8].apply(lambda x: str(x))
r15 = df[8].map(str)

arr = np.random.randint(0, 100, size=(100, 50))
r16 = np.sum(arr > 20)
r17 = np.nonzero(arr > 30)
r18 = np.where(arr > 40, 1, 0)


arr = np.random.randint(0, 100, size=(100, 50, 3))
r19 = np.sum(arr)
r20 = (arr*arr).sum(axis=-1)
r21 = (arr > 30).sum()


A = np.random.rand(10, 20, 50)
B = np.random.rand(50, 50, 2)
r22 = np.einsum('ijk,nkm->ijnm', A, B)
r23 = A.dot(B)


arr = np.random.normal(size=(10, 10))
r24 = np.vstack((arr, arr))
r25 = np.hstack((arr, arr))
r26 = np.hstack(arr)

arr = np.random.randint(10, size=10000)
r27 = np.column_stack((arr, arr))
r28 = np.array(range(10000))
r29 = np.atleast_2d(arr)
r30 = np.where(arr > 5)[0][0]
r31 = np.vectorize(oct, otypes=[object])(arr)


A = np.random.rand(1000, 200)
B = np.random.rand(200, 1000)
r32 = np.einsum('ij,jk->k', A, B)

df1 = pd.DataFrame(np.random.randn(1000, 10))
df2 = df1.copy()
df2.iloc[np.random.randint(100, size=1000), 0] = np.nan
r33 = df1.combine_first(df2)

df = pd.DataFrame(columns=['id', 'category'])
df['id'] = np.random.randint(3, size=100000)
df['category'] = np.random.choice(['a','b','c'], 100000)
r34 = pd.crosstab(df['id'], df['category'])
r35 = df.groupby('id')['category'].value_counts().unstack(fill_value=0)

r36 = np.empty(1000000); r36[:] = 0
r37 = np.empty(100000); r37[:] = 1
r38 = np.full(1000, 0)
r39 = np.full(1000, 1)

a = np.random.randint(100, size=(10, 10))
b = np.random.randint(100, size=(10, 10))
r40 = np.tensordot(a, b, axes=(1, 1))
r41 = np.tensordot(b, a, axes=(0, 0))

arr = np.arange(100).reshape(2, 50)
r42 = np.tile(arr, (5, 1, 1))

arr = np.arange(1200).reshape((-1, 3))
r43 = [np.linalg.norm(x) for x in arr]

dates = pd.date_range('2015', freq='min', periods=100)
dates = [date.strftime('%d %b %Y %H:%M:%S') for date in dates]
r44 = pd.to_datetime(dates)

arr = np.arange(100)
r45 = np.array([oct(x) for x in arr])

df = pd.DataFrame({'v1': np.random.choice(list('abcd'), 100), 'v2':np.random.randint(3, size=100)})
r46 = df.groupby(['v1','v2']).filter(lambda x: len(x) > 3)
r47 = df.query("v1 in ['a', 'b']")