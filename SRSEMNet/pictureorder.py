import os
root_path ="/media/wind/fangliang/实验记录/2022-6-9/111"
HR="/media/wind/fangliang/实验记录/2022-6-9/SEMPIC/SEMPIC_train_HR"
x2="/media/wind/fangliang/实验记录/2022-6-9/SEMPIC/SEMPIC_train_LR_bicbic/X2"
x3="/media/wind/fangliang/实验记录/2022-6-9/SEMPIC/SEMPIC_train_LR_bicbic/X3"
x4="/media/wind/fangliang/实验记录/2022-6-9/SEMPIC/SEMPIC_train_LR_bicbic/X4"
filelist = os.listdir(root_path)
i=0
for item in filelist:
    if item.endswith(".png"):
        src =os.path.join(os.path.abspath(root_path),item)
        if i%4 ==3:
            dst =os.path.join(os.path.abspath(HR),str(i/4)+'.png')
        elif i%4 ==2:
            dst = os.path.join(os.path.abspath(x2), str(i/4) + '.png')
        elif i%4 ==1:
            dst = os.path.join(os.path.abspath(x3), str(i/4) + '.png')
        else:
            dst = os.path.join(os.path.abspath(x4), str(i/4) + '.png')
    try:
        os.rename(src,dst)
        i+=1
        print('rename from %s to %s' %(src,dst))
    except:
        continue
print('ending......')