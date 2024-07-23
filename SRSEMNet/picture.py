import os
root_path ="/media/wind/fangliang/实验记录/2022-6-9/SEMPIC/SEMPIC_train_HR"
filelist = os.listdir(root_path)
i=0
for item in filelist:
    if item.endswith(".png"):
        src =os.path.join(os.path.abspath(root_path),item)
        dst =os.path.join(os.path.abspath(root_path),str(i)+'.tif')
    try:
        os.rename(src,dst)
        i+=1
        print('rename from %s to %s' %(src,dst))
    except:
        continue
print('ending......')