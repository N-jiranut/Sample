test_list = [[1,2],[3,4],[5,6]]
dele1, dele2 = 1,2
for id in range(len(test_list)):
    x, y = test_list[id]
    test_list[id]= (round((x-dele1)*100),round((y-dele2)*100))
print(test_list)