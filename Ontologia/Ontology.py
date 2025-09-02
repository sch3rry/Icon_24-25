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
        for prop in crop.HasSoilProperty:
            if hasattr(prop, factor_name) :
                value = getattr(prop, factor_name)
                print(value)
                if value:
                    value = value[0]
                    if min_val <= value <= max_val:
                        results.append((crop.name,factor_name, value))


    return results

def filter_by_climate(factor_name, min_val, max_val, onto):
    results = []
    try:
        Crop = onto.Crop
    except:
        print("Classe Crop non trovata nell'ontologia")
        return results

    for crop in Crop.instances():
        print(crop.name)

        for prop in crop.HasClimateProperty:
            if hasattr(prop, factor_name):
                value = getattr(prop, factor_name)
                if value:
                    value = value[0]
                    #print(f"{crop.name} = {factor_name}: {value}")
                    if min_val <= value <= max_val:
                        results.append((crop.name, factor_name, value))
                else:
                    print(f"{crop.name} = {factor_name}: no value")
            else:
                print(f"{crop.name} = {factor_name}: no valid property")
    return results

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
            factor = input("Inserisci fattore terreno (Nitrogen, Ph_value, Potassium, Phosphorus): ").strip()
            try:
                min_val = float(input("Valore minimo: "))
                max_val = float(input("Valore massimo: "))
                results = list_by_soil(factor, min_val, max_val, onto)
                if results:
                    for result in results:
                        print(f"Individuo: {result[0]}, {result[1]} = {result[2]}")
                else:
                    print("Nessun individuo trovato con i valori specificati.")
            except Exception as e:
                print(f"Errore nell'input: {e}")
        elif choice == "6":
            factor = input("Inserisci fattore climatico (temperature, humidity, rainfall): ").strip()
            try:
                min_val = float(input("Valore minimo: "))
                max_val = float(input("Valore massimo: "))
                results = filter_by_climate(factor, min_val, max_val, onto)
                if results:
                    for result in results:
                        print(f"Individuo: {result[0]}, {result[1]} = {result[2]}")
                else:
                    print("Nessun individuo trovato con i valori specificati.")
            except Exception as e:
                print(f"Errore nell'input: {e}")
        elif choice == "0":
            break
        else:
            print("Scelta non valida.")
