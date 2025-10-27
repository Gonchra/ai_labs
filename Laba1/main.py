import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("dataset/test.csv")

print(df.head(10))

missing_values = df.isnull().sum()
print(missing_values)

home_mode = df["HomePlanet"].mode()[0]
cryo_sleep_mode = df["CryoSleep"].mode()[0]
cabin_mode = df["Cabin"].mode()[0]
destination_mode = df["Destination"].mode()[0]
age_median = df["Age"].median()
vip_mode = df["VIP"].mode()[0]
room_service_median = df["RoomService"].median()
food_court_median = df["FoodCourt"].median()
shopping_mall_median = df["ShoppingMall"].median()
spa_median = df["Spa"].median()
vr_deck_median = df["VRDeck"].median()
name_mode = df["Name"].mode()[0]

df["HomePlanet"] = df["HomePlanet"].fillna(home_mode)
df["CryoSleep"] = df["CryoSleep"].fillna(cryo_sleep_mode)
df["Cabin"] = df["Cabin"].fillna(cabin_mode)
df["Destination"] = df["Destination"].fillna(destination_mode)
df["Age"] = df["Age"].fillna(age_median)
df["VIP"] = df["VIP"].fillna(vip_mode)
df["RoomService"] = df["RoomService"].fillna(room_service_median)
df["FoodCourt"] = df["FoodCourt"].fillna(food_court_median)
df["ShoppingMall"] = df["ShoppingMall"].fillna(shopping_mall_median)
df["Spa"] = df["Spa"].fillna(spa_median)
df["VRDeck"] = df["VRDeck"].fillna(vr_deck_median)
df["Name"] = df["Name"].fillna(name_mode)

missing_values = df.isnull().sum()
print(missing_values)

scaler = MinMaxScaler()

df["Age"] = scaler.fit_transform(df[["Age"]])
df["RoomService"] = scaler.fit_transform(df[["RoomService"]])
df["FoodCourt"] = scaler.fit_transform(df[["FoodCourt"]])
df["ShoppingMall"] = scaler.fit_transform(df[["ShoppingMall"]])
df["Spa"] = scaler.fit_transform(df[["Spa"]])
df["VRDeck"] = scaler.fit_transform(df[["VRDeck"]])

columns_to_transform = ["Destination", "HomePlanet"]

df = pd.get_dummies(df, columns=columns_to_transform)

columns_to_drop = ["Cabin", "Name", "PassengerId"]

for col in columns_to_drop:
    df = df.drop(col, axis="columns")

print(df.head(10))

df.to_csv("dataset/processed_test.csv", index=False)