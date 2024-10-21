import numpy as np
from json import dumps

webGraph = {
    "Yahoo": ["Yahoo", "Amazon"],
    "Amazon" : ["Yahoo", "Microsoft"],
    "Microsoft" : ["Amazon"],
    "C" : ["deadEndExample"],
    "deadEndExample": []
}

def computeMgMatrix(webGraph: dict) -> np.ndarray: 
    """
    Compute the stochastic matrix Mg (M for short)
    """
    webPages = list(webGraph)
    nbOfWebpage = len(webGraph)
    
    mgMatrix = np.zeros((nbOfWebpage, nbOfWebpage))
    
    for i, (_, linkedWebPages) in enumerate(webGraph.items()):
        row = np.zeros(nbOfWebpage)
        for linkedWebPage in linkedWebPages:
            row[webPages.index(linkedWebPage)] = 1 / len(linkedWebPages)
        mgMatrix[i] = row
    
    mgMatrix = mgMatrix.T
    
    return mgMatrix

def removeDeadEnds(webGraph: dict) -> dict:
    """
    Remove pages that don't link to any other page.
    """
    modifWebGraph = webGraph.copy()
    
    while True:
        deleted = False
        pagesToDelete = []
        
        for page, linkedPages in modifWebGraph.items():
            if len(linkedPages) == 0:
                pagesToDelete.append(page)
        
        if not pagesToDelete:
            break
        
        for page in pagesToDelete:
            del modifWebGraph[page]
            deleted = True
            
            for page2, linkedPages2 in modifWebGraph.items():
                modifWebGraph[page2] = [element for element in linkedPages2 if element != page]
        
        if not deleted:
            break

    return modifWebGraph


            
def randomSurferImproved(mgMatrix : np.ndarray, pagesScores: np.ndarray, nbOfWebpage: int, beta : float) -> np.ndarray:
    """
    Compute the next iteration of the PageRank vector without constructing the full A matrix.
    """
    betaMgPi = beta * np.dot(mgMatrix, pagesScores)
    
    uniformJump = np.full(nbOfWebpage, (1 - beta) / nbOfWebpage)
    
    newPagesScores = betaMgPi + uniformJump
    
    return newPagesScores

def hasConverged(oldPagesScores : np.array, newPagesScores : np.array, epsilon : int) -> bool:
    difference = np.abs(newPagesScores - oldPagesScores)
    normL1 = np.sum(difference)
    return normL1 < epsilon
    
    
def pageRank(webGraph : dict):
    beta = 0.85
    epsilon = 0.000001
    
    cleanedWebGraph = removeDeadEnds(webGraph)
    nbOfWebpage = len(cleanedWebGraph)
    
    mgMatrix = computeMgMatrix(cleanedWebGraph)
    pagesScores = np.full(shape = nbOfWebpage, fill_value=1/nbOfWebpage, dtype=float)
    
    while (True):
        newPagesScores = randomSurferImproved(mgMatrix, pagesScores, nbOfWebpage, beta)
        if (hasConverged(pagesScores, newPagesScores, epsilon)) : 
            print("Converged!")
            break
        pagesScores = newPagesScores
    return cleanedWebGraph, pagesScores

def main():
    cleanedWebGraph, pagesScores = pageRank(webGraph)
    pagesWebGraph = list(cleanedWebGraph)
    for page in range(len(pagesWebGraph)):
        print(f"Le site {pagesWebGraph[page]} a un score PageRank de {pagesScores[page]}.\n")
    
main()
    
