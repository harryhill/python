#coding: Utf-8
import MeCab

m = MeCab.Tagger('')

print(m.parse('すもももももももものうち'))
