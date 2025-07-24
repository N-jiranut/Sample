import pandas as pd
test_list = ["NI","GR"]
lista = [[1,2],[3,4],[5,6]]
df = pd.DataFrame(test_list)
df = df._append(lista,index=True)
print(df)