import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


path = ".\\data\\part2\\database\\"
nb_same_image = 5
list_files = glob.glob(path+ "*.jpg")

list_image = {}

for i, file in enumerate(list_files):
    file_name = file[len(path):-6]
    if(file_name not in list_image.keys()):
        list_image[file_name] = np.empty(nb_same_image, dtype=object)

for i, file in enumerate(list_files):
    file_name = file[len(path):-6]
    idx = int(file[-5])-1
    im = cv2.imread(file)
    list_image[file_name][idx] = im

def comparaison_ORB(query, train):
    training_image = cv2.cvtColor(train, cv2.COLOR_BGR2RGB)
    query_image = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    
    training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create()
    
    train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
    query_keypoints, query_descriptor = orb.detectAndCompute(query_gray, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(query_descriptor, train_descriptor)
    matches = sorted(matches, key = lambda x:x.distance)
        
    score = 0
    n = 0
    poids = 2
    for mat in matches:
        score += 1/(mat.distance + n*poids)
        n += 1
    #print(score)
    return score

 ## test des requetes  

path_query = ".\\data\\part2\\"

list_files_query = glob.glob(path_query+ "*.jpg")

for i, file in enumerate(list_files_query):
    file_name = file[len(path_query):-4]
    im_query = cv2.imread(file)
    print("\n" +file_name + ": \n")
    max_mean = 0
    score = np.empty((len(list_image)*5, 2), dtype=np.object)
    k = 0
    for name in list_image:
        moyenne = 0
        for image in list_image[name]:
            score[k, :] = [comparaison_ORB(im_query, image), name == file_name[:-6]]
            moyenne += score[k, 0]/5
            k+= 1
        #print("moyenne = ", str(moyenne/5))
        if moyenne>max_mean:
            max_mean = moyenne
            max_name = name
    score = score[score[:,0].argsort()]
    
    plt.plot(score[:,0], 'bo')
    plt.plot(np.where(score[:,1] == True)[0], score[np.where(score[:,1] == True)[0], 0], 'ro')
    plt.title(file_name)
    plt.savefig("./results/p2/ORB/result_"+file_name+".jpg")
    plt.show()
    print(file_name+ " est un(e) " + max_name + " ~ c'est " + ("vrai" if max_name == file_name[:-6] else "faux") + ".")
