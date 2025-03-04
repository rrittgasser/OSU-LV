hamCount = []
spamCount = []
endsWith = 0

with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as file:
    for linija in file:
        dijelovi = linija.strip().split('\t')
        if len(dijelovi) == 2:
            tip, tekst = dijelovi
            wordCount = len(tekst.split())

            if tip == 'ham':
                hamCount.append(wordCount)
            elif tip == 'spam':
                spamCount.append(wordCount)
                if tekst.strip().endswith('!'):
                    endsWith += 1

if hamCount:
    hamAverage = sum(hamCount) / len(hamCount)
else:
    hamAverage = 0

if spamCount:
    spamAverage = sum(spamCount) / len(spamCount)
else:
    spamAverage = 0

print("Ham average = ", round(hamAverage, 2))
print("Spam average = ", round(spamAverage, 2))
print("Ends with !: ", endsWith)
