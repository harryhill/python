#coding: Utf-8

import unittest
import pandas as pd
#import numpy as np
from sklearn import linear_model

#あくまで複数変数を単回帰するためのロジックということに注意
#重回帰や他の多変量解析を行う場合、dfから個々の変数を抜き出すロジックを立てなければならない（変数削除等）

dates = pd.date_range("20151111", periods=5)#df&ts用日付ラベルdates定義

s1 = pd.Series([2,3,4,5,6], index=dates)
s2 = pd.Series([10,20,30,40,50], index=dates)
s3 = pd.Series([100,200,300,400,500], index=dates)
kpi = pd.Series([1,2,3,4,5], index=dates)
df = pd.concat([kpi, s1, s2, s3], axis=1)#tsを結合してdfへ axis=0は縦結合なので変更しないで
df.columns = ["kpi","s1","s2","s3"]#変数名は結合時に反映されないので別途命名


print(df[["s1"]].as_matrix()) #重回帰分析(xは多変量)のため、説明変数は[[ ]]で取り出す必要あり
print(df["kpi"].as_matrix()) #被説明変数はdfそのまま取り出して平気

def singlelm(df):
	clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
	summary = [clf.fit(df[[i]].as_matrix(), df["kpi"].as_matrix()) for i in list(df.columns.values)[1:]]#右が被説明変数

	for j in summary:
		print("変数{0}の回帰分析結果".format(j))
		print("回帰係数：{0}".format(float(clf.coef_)))
		print("切片項：{0}".format(clf.intercept_))
		#r2 = clf.score(df[[j]].as_matrix(), df["kpi"].as_matrix())
		#print("決定係数R^2：{0}".format(r2))
		print( )
		#R-squareの別途計算にclf.score内の引数を再度正確に指定しないと決定係数-6.0とか出るので要注意

singlelm(df)



def lm(df):
	for i in list(df.columns.values)[1:]:
		clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
		clf.fit(df[[i]].as_matrix(), df["kpi"].as_matrix())#右が被説明変数
		print("変数{0}の回帰分析結果".format(i))

		print("回帰係数：{0}".format(float(clf.coef_)))
		print("切片項：{0}".format(clf.intercept_))
		r2 = clf.score(df[[i]].as_matrix(), df["kpi"].as_matrix())
		print("決定係数R^2：{0}".format(r2))
		print( )
		#R-squareの別途計算にclf.score内の引数を再度正確に指定しないと決定係数-6.0とか出るので要注意

lm(df)