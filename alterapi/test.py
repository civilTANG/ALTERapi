import alterapi
print("static mode")
x = alterapi.APIReplace('code.py', option= 'static' )
x.recommend()

print("dynamic mode")
x = alterapi.APIReplace('code.py', option= 'dynamic' )
x.recommend()
