# Loe võimalike lahenduste fail loendisse
with open('possible_words.txt', 'r') as f:
    possible_words = [line.strip() for line in f]

# Andmestruktuurid Wordle'i tagasisidest saadud informatsiooni hoidmiseks.
grey = set()
yellow = [set() for i in range(5)]
green = [None for j in range(5)]


# Funktsioonina defineeritud tingimus sõnade filtreerimiseks
def condition(x):
    # Esiteks ei tohi sõna tähtede hulgas olla halliks määratud tähti
    # ja kõik kollaste hulkades peaved olema sõnas olemas
    value = [grey.isdisjoint(set(x)), set.union(*yellow).issubset(set(x))]
    # Iga sõna iga koha kohta kontrollitakse, et kollaseks määratud täht ei ole vastaval kohal,
    # ja et kohtadel, millele on roheline täht määratud, on see täht.
    for i in range(5):
        value.append(x[i] not in yellow[i]
                     and (x[i] == green[i] or green[i] is None))
    return all(value)


# Programmi põhifunktsioon, mis võtab kasutajalt sisse sõna ning sõna tähtede värvid,
# lisab värvide põhjal tähed vastavasse loendisse/hulka ja koostab filtreeritud võimelike lahenduste loendi.
def guess_analysis():
    guess = input('Sisesta sõna: ')
    colors = input('sisesta värvid (h/k/r): ')

    i = 0
    for x in colors:
        if x == 'h':
            grey.add(guess[i])
        elif x == 'k':
            yellow[i].add(guess[i])
        elif x == 'r':
            green[i] = guess[i]
        i += 1

    filtered = [word for word in possible_words if condition(word)]
    print(filtered)


# Funktsiooni korratakse kuni kõik rohelised tähed on leitud.
while None in set(green):
    guess_analysis()
