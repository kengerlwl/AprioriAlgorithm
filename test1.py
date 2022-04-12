import json

f = open("para.json", 'r')
temp = json.loads(f.read())
print(temp)