s = ""
if s is None:
    print('\'\' == None')
else:
    print('\'\' != None')

if not s:
    print('\'\' == False')
else:
    print('\'\' != False')

if s == False:
    print('Can use \'\' == False')
else:
    print('Can not use \'\' == False')