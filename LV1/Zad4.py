wordCount = {}

file = open('song.txt')
for line in file:
    line = line.rstrip()
    words = line.split(" ")
    for word in words:
        if word not in wordCount:
            wordCount[word] = 1
        else:
            wordCount[word] += 1

file.close()

singleWords = []

for word in wordCount:
    if wordCount[word] == 1:
        singleWords.append(word)

print(f"Broj jedinstvenih riječi: {len(singleWords)}")
print("Jedinstvene riječi su:", ", ".join(singleWords))
