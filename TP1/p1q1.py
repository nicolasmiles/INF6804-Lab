import json
import cv2


with open("./data/part1/gt.json") as json_file:
    data = json.load(json_file)
    pictures = {}
    for i in data['annotations']:
        if i['image'] not in pictures.keys():
            path = r'.\\data\\part1\\images\\'+i['image'] + '.jpg'
            pictures[i['image']] = cv2.imread(path)
        if i['category_id'] == 8:
            pictures[i['image']] = cv2.rectangle(pictures[i['image']],(int(i['bbox'][0]),int(i['bbox'][1]))  , (int(i['bbox'][0]+i['bbox'][2]), int(i['bbox'][1]+i['bbox'][3])), (0,0,255), -1) 
        else:
            pictures[i['image']] = cv2.rectangle(pictures[i['image']],(int(i['bbox'][0]),int(i['bbox'][1]))  , (int(i['bbox'][0]+i['bbox'][2]), int(i['bbox'][1]+i['bbox'][3])), (255,0,0), 2) 
for im in pictures:
    #cv2.imshow(im, pictures[im])
    cv2.imwrite("./results/p1q1/"+im+".jpg", pictures[im])
cv2.waitKey(0)
cv2.destroyAllWindows()


        


