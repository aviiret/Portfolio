def leia_tase(t, v):
    if t < 15:
        t_tase = 0
    elif  15 <= t < 23:
        t_tase = 1
    elif 23 <= t < 30:
        t_tase = 2
    else:
        t_tase = 3
    if v < 20:
        v_tase = 0
    elif 20 <= v < 30:
        v_tase = 1
    elif 30 <= v < 50:
        v_tase = 2
    else:
        v_tase = 3
    if t_tase > v_tase:
        return t_tase
    else:
        return v_tase


fail = input('Sisesta failinimi: ')

with open(fail) as f:
    sademed = [int(rida) for rida in f]

tuul = 4
päev = 1
ohtlikke = 0
ohutuid = 0

for x in sademed:
    tase = leia_tase(tuul, x)
    if tase == 0:
        taseTekst = 'ilm ei ole ohtlik'
        ohutuid += 1
    else:
        taseTekst = 'ohutase: ' + str(tase)
    print(str(päev) + '. päev: vihm:', x, 'mm, tuul:', tuul, 'm/s,', taseTekst)
    päev += 1
    tuul += 5
    if tase == 3:
        ohtlikke += 1

print('3. ohutasemega päevi oli', ohtlikke, '. Ilm ei olnud ohtlik', ohutuid, 'päeval.')