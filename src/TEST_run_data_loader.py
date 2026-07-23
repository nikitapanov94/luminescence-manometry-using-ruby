from avantes_data_loader import load_folder

folder_input = input("Enter path to folder containing .txt files: ").strip()

all_data = load_folder(folder_input, max_rows=None)

print("\nDONE.")
print("Files loaded:", len(all_data))

if all_data:
    first_name = next(iter(all_data))
    d = all_data[first_name]

    print("First file:", first_name)

    print("x shape:", d["x"].shape)
    print("y shape:", d["y"].shape)

    x = d["x"]
    y = d["y"]

    print("\nFirst 3 rows of x and y:")
    for i in range(min(3, len(x))):
        print(f"{i}: x = {x[i]}, y = {y[i]}")

    print("\nLast 3 rows of x and y:")
    for i in range(max(0, len(x) - 3), len(x)):
        print(f"{i}: x = {x[i]}, y = {y[i]}")