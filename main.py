from Kb.menu import menu_kb
from Ontologia.Ontology import view_ontology
import Modelli.prediction
import os

def main_menu():
    while True:
        os.system('cls')
        print("\n=== MAIN MENU ===")
        print("1. Consultare l'ontologia")
        print("2. Consultare la knowledge base")
        print("3. Effettuare una predizione")
        print("0. Esci")

        choice = input("\nSeleziona: ")
        if choice == "1":
            view_ontology()
        elif choice == "2":
            menu_kb()
        elif choice == "3":
            Modelli.prediction.make_prediction()
        elif choice == "0":
            print("Exiting...")
            break
    else:
        print("Inserire una opzione valida")
if __name__ == "__main__":
    main_menu()