def GetTotal():
    total = numberOfHours*hourWage
    return total
    
    
numberOfHours = float(input('Radni sati: \n'))
hourWage=float(input('eura/h: \n'))
total =GetTotal()
print("Ukupno", total, "eura")



