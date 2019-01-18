# based on the paper entitled "Optimal Capacitor Placement for Loss Reduction"

# before running this program add your python address, for example C:\Python27, to environement variable.
# in order to run this program you need some libraries to be installed on your python, namely "deap" from the following address:
# https://pypi.python.org/pypi/deap/
# take care that the library matches with your python version

import os,sys

# initialization of PSSE.
# PSSE initialization
psspath=r'C:\Program Files (x86)\PTI\PSSEXplore34\PSSPY27'   # change to the address of your PSSBIN directory on your computer
sys.path.append(psspath)
os.environ['PATH'] += ';' + psspath
psspath=r'C:\Program Files (x86)\PTI\PSSEXplore34\PSSBIN'   # change to the address of your PSSBIN directory on your computer
sys.path.append(psspath)
os.environ['PATH'] += ';' + psspath

import psspy
import redirect
import random

redirect.psse2py()
psspy.psseinit(10000)
casestudy=(r'martin_2013_nzar.sav')  # change the address and filename to the network that you want to apply the algorithm
psspy.case(casestudy)   # load psse model defined by casestudy. since ardat function give areal losses we need these IDs to sum up all areal losses to find overall network loss
ierr, areas = psspy.aareaint(-1, 1, 'NUMBER') # id of areas in the network.
psspy.fdns(OPTIONS1=0,OPTIONS5=0,OPTIONS6=1)    # run power flow in psse model using decoupled newton method
PLOSS1=0
for i in areas[0]:  # evaluating sum of losses in all areas, before compensation
    ierr, area_loss=psspy.ardat(iar=i,string='LOSS')
    PLOSS1 =PLOSS1+area_loss.real

ierr, vpu1 = psspy.abusreal(-1, string="PU")    # voltage at all buses, before compensation
vpu01=vpu1[0]

ierr, busidx = psspy.abusint(-1, string='NUMBER') # find All buses
busidx0=[]
nbus=len(busidx0)   # No. of all buses
ierr, PVidx = psspy.agenbusint(-1, 1, 'NUMBER') # find PV buses
for idx in busidx[0]:        # find PQ buses by removing PV buses from All buses
    if idx not in PVidx[0]:
        busidx0.append(idx)

Vmin=0.98       # here you can define your desired Vmin and Vmax
Vmax=1.02

# next three line import methods from deap class to define genetic algorithm. to find out more go to deap library documentation
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

D=6 # number of steps, D in the main article.
Q0=10 # MVAR step size, Qc in the main article.
seq= range(0,D*Q0,Q0)   # sequance of possible values for reactive power
N_cap=len(busidx0) # No. of Capacitors to be located over the network. here I have considered shunts on all PQ buses. value 0 for a capacitor, shows that there is no shunt connected to it.

toolbox = base.Toolbox()
toolbox.register("attr_float", random.choice,seq)
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, n=N_cap)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual): # defining the objective function
    psspy.case(casestudy)
    for i in range(len(busidx0)):
        ierr = psspy.shunt_data(busidx0[i], ID=1, INTGAR=1, REALAR1=0,REALAR2=individual[i])
    psspy.fdns(OPTIONS1=0,OPTIONS5=0,OPTIONS6=1)
    
    PLOSS=0
    for i in areas[0]:  # evaluating sum of losses in all areas
        ierr, area_loss=psspy.ardat(iar=i,string='LOSS')
        PLOSS =PLOSS+area_loss.real

    ierr, vpu = psspy.abusreal(-1, string="PU")
    vpu0=vpu[0]
    JV=0;
    for i in range(nbus):
        if busidx[0][i] in busidx0:
            JV=JV+min(0,Vmin-vpu0[i])**2+max(0,vpu0[i]-Vmax)**2 # Adding voltage limit, Vmin < V < Vmax, as a part of objective function
    W1 , W2 = 1, 10
    J=W1*PLOSS+W2*JV; # Objective function
    # print("  Loss %s" % PLOSS)

    return J,

# Operator registering: here you can define algorithms for crossover, mutation and selection
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)
    
    pop = toolbox.population(n=10)
    CXPB, MUTPB, NGEN = 0.7, 0.2, 20    # Crossover probability, Mutation Probability, No. of generations
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()

# after compensation
PLOSS2=0
for i in areas[0]:  # evaluating sum of losses in all areas
    ierr, area_loss=psspy.ardat(iar=i,string='LOSS')
    PLOSS2 =PLOSS2+area_loss.real

ierr, vpu2 = psspy.abusreal(-1, string="PU")
vpu02=vpu2[0]

ierr = psspy.save(r'martin_2018_nzar_compensated.sav')

print ("Ploss before %s" %PLOSS1)
print ("Ploss after %s" %PLOSS2)
print ("Voltage before %s" %vpu01)
print ("Voltage after %s" %vpu02)



    
