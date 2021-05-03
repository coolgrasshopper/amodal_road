import os
import cv2
import json
import numpy as np

def add_to_dict(data, itr, key, count):
    points=data[itr]["data"]
    #print(points)
    file_bbs[key]=points

files = [ f.path for f in os.scandir("amodal2/") if f.is_dir() ]
for file in files:
    source_folder = file
    if not os.path.isdir(source_folder+"/mask/"):
        os.mkdir(source_folder+"/mask/")
    #json_path = "1576607257725640.json"                     # Relative to root directory
    json_files = [pos_json for pos_json in os.listdir(source_folder+"/annotations/") if pos_json.endswith('.txt')]
    print(json_files)  # for me this prints ['foo.json']

    count = 0                                           # Count of total images saved
    MASK_WIDTH = 1920				    # Dimensions should match those of ground truth image
    MASK_HEIGHT = 1208

    # Extract X and Y coordinates if available and update dictionary
    # Read JSON file
    for json_path in json_files:
        file_bbs = {}
        with open(source_folder+"/annotations/"+json_path) as f:
          data = json.load(f)
          #print(data)

        sub_count=0
        for itr in range(len(data)):
            key = str(data[itr]["class"])+str(itr)
            add_to_dict(data, itr, key, sub_count)



        print("\nDict size: ", len(file_bbs))
        print(file_bbs)

        # For each entry in dictionary, generate mask and save in correponding
        # folder
        mask = np.zeros((MASK_HEIGHT, MASK_WIDTH,3),dtype=np.uint8)
        #mask.fill((128))
        for itr in file_bbs:
            try:
                arr = np.array(file_bbs[itr])
            except:
                print("Not found:", itr)
                continue

            cv2.fillPoly(mask, [arr], color=(128,64,128))

        label_seg = np.zeros((MASK_HEIGHT, MASK_WIDTH),dtype=np.uint8)
        #label_seg.fill(128)
        b = np.array([128,64,128])
        c=np.array([0,0,0])
        #print((mask==b).all(axis=2))
        indices_list=np.where(np.any(mask!=c,axis=-1))

        #b = np.array([255,255,255])
        label_seg[indices_list]=1

        #label_seg[(mask==b).all(axis=2)] = 1
        #label_seg[(mask==c).all(axis=2)] = 0


        count += 1
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file+"/mask/"+json_path.split(".")[0]+".png", label_seg)


    print("Images saved:", count)
