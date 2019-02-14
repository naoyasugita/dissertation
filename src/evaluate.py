import pandas as pd
import glob

path = "./dataset/evaluation/*.csv"

csv_arr = glob.glob(path)
df = []
for csv in csv_arr:
    df.append(pd.read_csv(csv))


# _type : Quality, Completeness, Correctness
# describe : max, min, mean...
def computed_data(_type, describe):
    print(_type + ", " + describe)
    for _file, name in zip(df, csv_arr):
        name = name.split("/")[-1]
        print(name + " : " + '%.3f' % (_file.describe()[_type][describe]))
    print("=" * 24)


if __name__ == "__main__":
    type_arr = ["Quality", "Completeness", "Correctness"]
    describe_arr = ["mean", "max"]
    for describe in describe_arr:
        for _type in type_arr:
            computed_data(_type, describe)
