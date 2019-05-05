# SOURCE https://github.com/Sentdex/pygta5/blob/master/Tutorial%20Codes/Part%2014-15/getkeys.py

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
	keyList.append(char)

def get_key():
	keys = []
	for key in keyList:
		if wapi.GetAsyncKeyState(ord(key)):
			keys.append(key)
	return keys