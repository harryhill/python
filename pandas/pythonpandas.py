#coding: Utf-8

import pandas as pd

dates = pd.date_range("20151111", periods=5)#df&ts用日付ラベルdates定義

id = pd.Series(["111","112","111","113","113"], index=dates)
s1 = pd.Series([2,3,4,5,6], index=dates)
s2 = pd.Series([10,20,30,40,50], index=dates)
s3 = pd.Series([100,200,300,400,500], index=dates)
kpi = pd.Series([1,2,3,4,5], index=dates)

df = pd.concat([id, kpi, s1, s2, s3], axis=1)#tsを結合してdfへ axis=0は縦結合(bind)
df.columns = ["id","kpi","s1","s2","s3"]#変数名は結合時に反映されないので別途命名

print(df)

print(df.describe())#基礎統計量算出

print(df.ix[:,"s2"])#slice
print(df["s2"])#slice


df_grouped = df.groupby("id")
print(df_grouped[["kpi", "s1", "s2", "s3"]].sum())





dfjoin = pd.concat([df, df2], join = 'inner', keys = id)#inner or outer


x = pd.Series([1,2,3])
x.to_csv(“/Users/python/data.csv”)# to write csv file

y = pd.read_csv(“/Users/python/data.csv”)# to read csv file

