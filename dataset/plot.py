#analisi dei dati

import os
import matplotlib.pyplot as plt
import seaborn as sns


def plotting(df, save_dir):


    os.makedirs(save_dir, exist_ok=True)

    for col in df.drop(columns=['label']).columns:
        #istogramma
        plt.figure(figsize=(6, 6))
        plt.hist(df[col].dropna(), bins= 6, color='skyblue', edgecolor='black')
        plt.title(col)
        plt.legend()
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"hist_{col}.png"))

        plt.show()
        plt.close()

        #boxplot
        plt.figure(figsize=(6, 3))
        plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"boxplot_{col}.png"))
        plt.show()
        plt.close()

    #pie chart
    plt.figure(figsize=(8,8))
    crop_counts = df['label'].value_counts()
    plt.pie(crop_counts, autopct='%1.1f%%', startangle=140)
    plt.title('Distribuzione delle colture')
    plt.savefig(os.path.join(save_dir, "pie_chart_labels.png"))
    plt.show()
    plt.close()




    #heatmap
    corr_matrix = df.drop(columns=['label']).corr()
    plt.figure(figsize=(10,8))
    plt.title('Correlation Matrix')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig(os.path.join(save_dir, "heatmap_corr.png"))
    plt.show()
    plt.close()

    print(f"Grafici salvati nella cartella: '{save_dir}'")