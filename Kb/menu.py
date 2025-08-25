import pyswip as ps
import os

def menu_kb():
    prolog = ps.Prolog()
    prolog.consult('Kb/Kb.pl')
    while True:
        os.system('cls')
        print(
            'Select one of the following options:\n'
            '1. Semina in base al Mese\n'
            '2. Scelta della semina in base al Budget e terreno disponibili\n'
            '3. Scelta della semina per condizioni climatiche\n'
            '4. Colture simili a quella data\n'
            '5. Coltura consigliata in base ai parametri\n'
            '0. Torna indietro\n'
        )


        choice = input("|____: ")

        match choice:
            case "0":
                return
            case "1":
                mese = input('Inserisci il mese (es. january): ')
                crops = list(prolog.query(f"suggest_crops_by_month('{mese}', Crops)"))
                if crops:
                    print("Colture suggerite:", crops[0]['Crops'])
                else:
                    print("Nessuna coltura adatta trovata.")
            case "2":
                acri = int(input("Inserisci numero di acri: "))
                budget = int(input("Inserisci budget disponibile: "))
                crops = list(prolog.query(f"sowing_economical_info({acri}, {budget}, Crop)"))
                if crops:
                    print("Colture possibili:", [c['Crop'] for c in crops])
                    min_crop = list(prolog.query(f"min_cost_crop({acri}, {budget}, BestCrop)"))
                    if min_crop:
                        print("Coltura con costo minimo:", min_crop[0]['BestCrop'])
                else:
                    print("Nessuna coltura possibile con il budget dato.")
            case "3":
                ph = input("Inserisci zona PH (acido/neutro/basico): ")
                umid = input("Inserisci zona umidità (bassa/media/alta): ")
                rain = input("Inserisci zona pioggia (arida/standard/piovosa): ")
                crops = list(prolog.query(f"sowing_zone_info('{ph}', '{umid}', '{rain}', Crop)"))
                if crops:
                    print("Colture consigliate:", [c['Crop'] for c in crops])
                else:
                    print("Nessuna coltura corrisponde alle condizioni climatiche.")

            case "4":
                ref = input("Inserisci il nome della coltura di riferimento: ")
                sim = list(prolog.query(f"most_sim('{ref}', SimCrop)"))
                if sim:
                    print(f"Coltura più simile a {ref}:", sim[0]['SimCrop'])
                else:
                    print("Nessuna coltura simile trovata.")

            case "5":
                acri = int(input("Inserisci numero di acri: "))
                budget = int(input("Inserisci budget disponibile: "))
                ph = input("Inserisci zona PH (acido/neutro/basico): ")
                umid = input("Inserisci zona umidità (bassa/media/alta): ")
                rain = input("Inserisci zona pioggia (arida/standard/piovosa): ")
                crops = list(prolog.query(f"recommended_crop({acri}, {budget}, '{ph}', '{umid}', '{rain}', Crop)"))
                if crops:
                    print("Coltura consigliata:", [c['Crop'] for c in crops])
                else:
                    print("Nessuna coltura consigliata con le condizioni e budget dati.")
