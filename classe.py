#https://github.com/hotbread213/createClass.git

''' Programme pour placé les élèves dans la classe '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import json
import pprint as pp
import itertools
import random
import numba
import math

class OrganisationDeLaClasse():

    def __init__(self):
        self.format_x = 7
        self.format_y = 5

    def createGrid(self):
        # Ici, on crée une grid rempli de NAN
        
        self.classe_grid = np.zeros(shape = (self.format_x, self.format_y), dtype='U40')

    def getlistStudent(self):

        self.list_Eleve = [
            'Arseneault, Sarah-Maude',
            'Polichronis, Erika ',
            'Beauregard, Mia',
            'Beauvoir, David',
            'Bertrand, Jakob',
            'Bisaillon, Gabrielle',
            'Bouchard, Marie-Anne',
            'Cabral, Raphaël',
            'Cardinal-Laporte, Zacharie',
            'Chénier Rocha, Amélie',
            'Couture, Noémie',
            'Dallaire, Janin',
            'Delisle, Antoine',
            'Dubé-Coutu, Philippe',
            'Dubreuil, Jeanne',
            'Forbes, Annabella',
            'Frenette Cyr, Charles',
            'Gautreau-Miron, Arianne',
            'Gelez, Philippe',
            'Hébert, Thomas',
            'Kabambi-Loshi, Theresa',
            'Lavallée Brillant, Eugénie',
            'Leclair-Lapointe, Amélya',
            'Limon Lord, Mathieu',
            'Mailloux, Isabelle',
            'Martel, Patrick',
            'Migneault, Gabrielle',
            'Morissette, Tristan',
            'Sirois, Jérémie',
            'St-Louis, Samuel',
            'Stoliarov, Lior'
        ]

        self.getInFrontClass()
        self.getInBackClass()
        self.getMaxDistance()


    def getInFrontClass(self):

        self.listFront = [
            'Frenette Cyr, Charles',
            'Chénier Rocha, Amélie',
            'Polichronis, Erika',
            'Bouchard, Marie-Anne',
            'Sirois, Jérémie',
            'Forbes, Annabella',
            'Bisaillon, Gabrielle'
        ]

    def getInBackClass(self):

        self.listBack = [
        ]

    def getMaxDistance(self):

        self.listDist = [
            ('Dallaire, Janin', 'Bertrand, Jakob'),
            ('Dallaire, Janin', 'Dubé-Coutu, Philippe'),
            ('Dallaire, Janin', 'Stoliarov, Lior'),
            ('Dallaire, Janin', 'Limon Lord, Mathieu'),
            
            ('Dubé-Coutu, Philippe', 'Bertrand, Jakob'),
            ('Dubé-Coutu, Philippe', 'Stoliarov, Lior'),
            ('Dubé-Coutu, Philippe', 'Limon Lord, Mathieu'),

            ('Bertrand, Jakob', 'Stoliarov, Lior'),
            ('Bertrand, Jakob', 'Limon Lord, Mathieu'),

            ('Limon Lord, Mathieu', 'Stoliarov, Lior')
        ]

    def extendListStudent(self):
        N = self.list_Eleve

        # Ici, combiner la liste des eleves avec des faux eleves si la place n'est pas utilisée
        place = self.format_x*self.format_y

        self.list = N

        if len(self.list) < place:
            placeNAN = place - len(self.list)
            for i in range(placeNAN):
                self.list.append('NAN')

    def initialiseInGrip(self):

        ''' Place les etudiants au hazard dans la classe '''

        liste1 = range(self.format_x)
        liste2 = range(self.format_y)
        random.shuffle(self.list, random.random)

        i=0
        for key1 in liste1:
            for key2 in liste2:
                self.classe_grid[key1, key2] = self.list[i]
                i+=1

    @numba.jit
    def findIndex(self, key):
        index = np.where(self.classe_grid == key)
        return index

    @numba.jit
    def getDistanceMatrix(self, Mult):
        self.getlistStudent()
        dist = pd.DataFrame(index = self.list_Eleve, columns = self.list_Eleve)

        i, j = 0, 0
        for key1 in self.list_Eleve:
            for key2 in self.list_Eleve:
                if (key1, key2) in self.listDist:
                    index1 = self.findIndex(key1)
                    index2 = self.findIndex(key2)
                    dist.loc[key1, key2] = Mult*getEuclidianDist(index1, index2)
                else:
                    dist.loc[key1, key2] = 0

                j+=1
            i+=1

        return dist

    def constraintMatrixFront(self, Mult):
        self.getlistStudent()
        dist = pd.Series(index = self.list_Eleve)

        for key1 in self.list_Eleve:
            index1 = np.where(self.classe_grid == key1)

            mult=0
            if key1 in self.listFront:
                mult = Mult

            dist.loc[key1] = mult*(self.format_y - index1[0])

        return dist

    def constraintMatrixBack(self, Mult):
        self.getlistStudent()
        dist = pd.Series(index = self.list_Eleve)

        for key1 in self.list_Eleve:
            index1 = np.where(self.classe_grid == key1)

            mult=0
            if key1 in self.listBack:
                mult = Mult

            dist.loc[key1] = mult*(self.format_y)

        return dist

    @numba.jit
    def getAllMatrix(self):
        
        Fitness = self.getDistanceMatrix(2).sum().sum() + self.constraintMatrixFront(2).sum() #+ self.constraintMatrixBack(1).sum()

        return Fitness

    def geneticAlgo(self):

        N = 100

        Fitness = pd.DataFrame(index = range(N), columns = ['Fitness'])

        for i in range(N):

            # Ici, on doit prendre dans la population
            # Alors, on prend la population de 50 et on fait des bébés
            # des agents au hazard ( avec probabilité relier au fitness )

            if i==0:
                fitnessMoins1 = self.getAllMatrix()
            else:
                fitnessMoins1 = fitness

            population = []
            fitnessPop = []
            sizepop = 10
            for j in range(sizepop):
                self.initialiseInGrip()
                population.append(self.classe_grid)
                fitnessPop.append(self.getAllMatrix())

            prob = fitnessPop/sum(fitnessPop)

            self.population = population
            a = np.random.choice(range(sizepop), 2, True, prob)

            # Ici, c'est la mutation ( qui arrive avec une probabilité de )
            ii1 = random.randint(0, self.format_x-1)
            jj1 = random.randint(0, self.format_y-1)

            ii2 = random.randint(0, self.format_x-1)
            jj2 = random.randint(0, self.format_y-1)

            a = self.classe_grid[ii1, jj1]
            b = self.classe_grid[ii2, jj2]

            self.classe_grid[ii1, jj1] = b
            self.classe_grid[ii2, jj2] = a

            fitness = self.getAllMatrix()

            if fitnessMoins1 > fitness:
                self.classe_grid[ii1, jj1] = a
                self.classe_grid[ii2, jj2] = b
                fitness = fitnessMoins1

            Fitness.loc[i] = fitness
            self.Fitness = Fitness
            self.Classe = pd.DataFrame(self.classe_grid, index = range(self.format_x), columns = range(self.format_y)) 
            print(i)
            print(Fitness)

@numba.jit
def getEuclidianDist(x, y):

    assert len(x) == len(y)

    dist = 0
    for i in range(len(x)):
        dist += (x[i] - y[i])**2

    dist = np.sqrt(dist)

    return dist

if __name__ == '__main__':
    pf = OrganisationDeLaClasse()
    pf.createGrid()
    pf.getlistStudent()
    pf.extendListStudent()
    pf.initialiseInGrip()
    pf.geneticAlgo()

    Excel_to_Write = "Classe" + ".xlsx"
    writer = pd.ExcelWriter(Excel_to_Write, engine='openpyxl')
    pf.Classe.to_excel(writer, 'classe')
    pf.Fitness.to_excel(writer, 'objectif')

    writer.save()

    import os
    os.system('start ' + Excel_to_Write)

    pf.Fitness.plot()
    plt.show()