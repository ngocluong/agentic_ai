import pandas as pd

df = pd.read_csv('data/ecommerce.csv')
df['revenue'] = df['price'] * df['quantity']

def top_user():
    return df.groupby('user_id')['revenue'].sum().nlargest(3)

def revenue_by_country():
  return df.groupby('country')['revenue'].sum()

def agent(q):
  if 'country' in q:
    print(revenue_by_country())
  elif 'top user' in q:
    print(top_user())

question = input("what is your question: ")
print(agent(question))