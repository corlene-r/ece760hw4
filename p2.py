import io 
import os
import re
import numpy as np

np.set_printoptions(precision=4, suppress=True)

############################################################
#                    Helper Functions                      #
############################################################
def countChars(filename): 
    fopen = io.open(filename, 'r')
    str = fopen.read()
    str = str.lower()
    counts = np.zeros((27))
    i = 0
    for a in "abcdefghijklmnopqrstuvwxyz ":
        for c in str:
            if c == a:
                counts[i] = counts[i] + 1
        i = i + 1
    return counts
def logProbforLang(counts, probs): 
    logprob = 0
    for i in range(counts.shape[0]): 
        logprob = logprob + counts[i] * np.log(probs[i])
    return logprob
def predictLang(vecprobs, langprobs):
    maxProb = -np.inf
    lang = 0
    for j in range(3): 
        postprob = vecprobs[j] + np.log(langprobs[j])
        if postprob > maxProb:
            lang = j
            maxProb = postprob
    return lang
def randomizeFileOrder(filename): 
    fopen = io.open(filename, 'r')
    s = fopen.read()
    s = s.lower()
    idxs = np.random.permutation(len(s))
    s = [s[i] for i in idxs]
    return s

############################################################
#                        Problems                          #
############################################################
def p1():
    print("\n--------------------Problem 1--------------------")
    files = os.listdir("languageID")
    langs = [re.compile(r'e*.txt'), re.compile(r'j*.txt'), re.compile(r's*.txt')]
    langprobs = np.zeros(len(langs))
    for f in files: 
        for i in range(3): 
            if langs[i].search(f):
                langprobs[i] = langprobs[i] + 1

    denom = sum(langprobs) + 0.5 * len(langprobs)
    print("Probability of English file:\t",  (langprobs[0] + 0.5) / denom)
    print("Probability of Spanish file:\t",  (langprobs[1] + 0.5) / denom)
    print("Probability of Japanese file:\t", (langprobs[2] + 0.5) / denom)
    return langprobs
def p2and3(): 
    print("\n-----------------Problem 2 and 3-----------------")
    allProbs = np.zeros((27, 3))
    for j in range(3):
        lang = "ejs"[j]
        counts = np.zeros((27))
        probs = counts
        for i in range(10): 
            counts = counts + countChars("languageID/"+lang+str(i)+".txt")
        denom = np.sum(counts) + 0.5*counts.shape[0]
        for i in range(counts.shape[0]):
            probs[i] = (counts[i] + 0.5) / denom
        print("\tPrior Probabilities for language", lang, ":\n", probs)
        allProbs[:, j] = probs
    return allProbs
def p4():
    print("\n--------------------Problem 4--------------------")
    counts = countChars("languageID/e10.txt")
    print(counts)
def p5(probs):
    print("\n--------------------Problem 5--------------------")
    counts = countChars("languageID/e10.txt")
    vecprobs = np.zeros((3))
    for j in range(probs.shape[1]):
        vecprob = logProbforLang(counts, probs[:, j])
        print("Log probability for vector given language " + "ejs"[j] + ":", vecprob)
        vecprobs[j] = vecprob
    return vecprobs
def p6(langprobs, vecprobs): 
    print("\n--------------------Problem 6--------------------")
    for j in range(3): 
        posteriorprob = vecprobs[j] + np.log(langprobs[j])
        print("Log of Bayes Posterior for Language " + "ejs"[j] + ":", posteriorprob)
def p7(langprobs, priorprobs):
    print("\n--------------------Problem 7--------------------")
    confusionMat = np.zeros((3, 3))
    for j in range(3):
        lang = "ejs"[j]
        for i in range(10, 20):
            counts = countChars("languageID/" + lang + str(i) + ".txt")
            logprobs = np.zeros((3,1))
            for k in range(3):
                logprobs[k] = logProbforLang(counts, priorprobs[:, k])
            k = predictLang(logprobs, langprobs)
            confusionMat[k, j] = confusionMat[k, j] + 1
    print("Generated confusion matrix for language classifier:")
    print("  e    j    s")
    print(confusionMat)
def p8(langprobs, priorprobs):
    print("\n--------------------Problem 8--------------------")
    filename = "j18.txt"
    print("Using file " + filename)
    str = randomizeFileOrder("languageID/" + filename)
    counts = np.zeros((27))
    i = 0
    for a in "abcdefghijklmnopqrstuvwxyz ":
        for c in str:
            if c == a:
                counts[i] = counts[i] + 1
        i = i + 1
    for j in range(3): 
        priorprob = logProbforLang(counts, priorprobs[:, j])
        posteriorprob = priorprob + np.log(langprobs[j])
        print("\tLog Probability for Language " + "ejs"[j] + ":", posteriorprob)

langprobs = p1()
priorprobs = p2and3()
p4()
vecprobs = p5(priorprobs)
p6(langprobs, vecprobs)
p7(langprobs, priorprobs)
p8(langprobs, priorprobs)