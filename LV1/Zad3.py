UserInput = []
while(1):
    broj = input("Unesite broj ili Done: ")
    if broj == "Done":
        break
    try:
        broj = int(broj)
        UserInput.append(broj)
    except ValueError:
        print("Unešeni znak nije broj!")

print("Unijeli ste ", len(UserInput), "elemenata.")
print("Srednja vrijednost unešenih brojeva je: ", sum(UserInput) / len(UserInput))
print("Maksimalna vrijednost je: ", max(UserInput))
print("Minimalna vrijednost je: ", min(UserInput))
print("Sortirana lista:", sorted(UserInput))   

