from owlready2 import *
import os

# Costruisci il percorso completo al file Ontology.rdf
onto = get_ontology("Ontologia/Ontology.rdf").load()

def list_classes():
    return [cls.name for cls in onto.classes()]

def list_data_properties():
    return [dp.name for dp in onto.properties()]

def list_object_properties():
    return [op.name for op in onto.object_properties()]

def list_individuals():
    return [ind.name for ind in onto.individuals()]

def list_by_soil(factor_name, min_val, max_val, onto):
    results = []
    try:
        Crop = onto.Crop
    except:
        print("Classe Crop non trovata nell'ontologia")
        return results

    for crop in Crop.instances():
        for prop in crop.HasSoilProperty():
            if hasattr(prop, factor_name) and prop.__getattr__(factor_name):
                value = getattr(crop, factor_name)[0]
            if min_val <= value <= max_val:
                results.append((crop.name, value))

    return results

def filter_by_climate(factor_name, min_val, max_val, onto):
    results = []
    try:
        Crop = onto.Crop
    except:
        print("Classe Crop non trovata nell'ontologia")
        return results

    for crop in Crop.instances():
        for prop in crop.HasClimateProperty:
            if hasattr(prop, factor_name) and prop.__getattr__(factor_name):
                value = getattr(prop, factor_name)[0]
            if min_val <= value <= max_val:
                results.append((crop.name, crop.crop_name[0], factor_name, value))
    if results.__len__() == 0:
        print(f"nessun individuo ha {factor_name} compreso in {max_val} e {min_val}")
    else:
        print(f"{results}")
    #return results

def query_6(factor_name, min, max, onto):
    result = []
    try:
        for crop in onto.Crop.instances():
            for environmental_factor in crop.hasClimateProperty:
                attribute_values = getattr(environmental_factor, factor_name)
                if min <= attribute_values[0] <= max:
                    result.append(crop)
        if result.__len__() == 0:
            print(f'nessuno degli individui ha {factor_name} tra {min} e {max}')
        else:
            print(f'Risultati Crop con {factor_name} tra {min} e {max}:{result}')
    except:
        print('nome del fattore inserito errato')

def view_ontology():
    while True:
        print("\n--- MENU ONTOLOGIA ---")
        print("1. Lista classi")
        print("2. Lista data property")
        print("3. Lista object property")
        print("4. Lista di individui")
        print("5. Filtra in base ad un valore terreno (azoto, ph, potassio, fosforo)")
        print("6. Filtra in base al clima (umidità, pioggia ,temperatura)")
        print("0. Torna indietro")

        choice = input("Scelta: ")

        if choice == "1":
            print(list_classes())
        elif choice == "2":
            print(list_data_properties())
        elif choice == "3":
            print(list_object_properties())
        elif choice == "4":
            print(list_individuals())
        elif choice == "5":
            factor = input("Inserisci fattore climatico (Nitrogen, Ph_value, Potassium, Phosphorus): ")
            try:
                min_val = int(input("Valore minimo: "))
                max_val = int(input("Valore massimo: "))
                print(list_by_soil(factor, min_val, max_val, onto))
            except:
                print("Input non valido")
        elif choice == "6":
            factor = input("Inserisci fattore climatico (temperature, humidity, rainfall): ")

            min_val = int(input("Valore minimo: "))
            max_val = int(input("Valore massimo: "))
            query_6(factor, min_val, max_val, onto)
            #list_by_soil(factor, min_val, max_val, onto)
            #print(list_by_soil(factor, min_val, max_val, onto))

        elif choice == "0":
            break
        else:
            print("Scelta non valida.")
