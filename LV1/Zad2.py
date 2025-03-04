try:
    number = float(input("Upišite broj: \n"))
    if number>=0.9 and number <=1: 
        print("Ocjena pripada kategoriji A.")
    elif number>=0.8 and number <1:
        print("Ocjena pripada kategoriji B.")
    elif number>=0.7 and number <1:
        print("Ocjena pripada kategoriji C.")
    elif number>=0.6 and number <1:
        print("Ocjena pripada kategoriji D.")
    elif number>0 and number<=1:
        print("Ocjena pripada kategoriji F.")
    else: raise Exception ("Broj nije unutar intervala 0-1")
except  ValueError:
    print("Upisani znak nije broj. Upišite broj!")

except Exception as e:
    print(e)