from subprocess import call
import subprocess
import multiprocessing as mp
import os
import json
def call_cmd(obj_name,id,gpu_id):
    for scale in [0.06,0.08,0.10,0.12]:
        cmd=f"python timing.py --Trail_id {id} --obj_name {obj_name} --obj_scale {scale}"
        print("Start!",cmd)
        ret=call(cmd,shell=True,timeout=100)  
        print("Done!",id)
    return ret

def main(gpu_id,obj_name_list):
    save_path="../results"
    real_obj_name_list=[]
    for obj_name in obj_name_list:
        tmp_path=os.path.join(save_path,obj_name)
        for file in ["scale006","scale008","scale010","scale012"]:
            if(not os.path.exists(os.path.join(tmp_path,file))):
                print("obj_name",obj_name,"lack",file)
                real_obj_name_list.append(obj_name)
                break
    print("real_obj_name_list",len(real_obj_name_list))
    obj_name_list=real_obj_name_list
    total_num=len(obj_name_list)
    test_num=total_num
    for i in range(test_num):
        call_cmd(obj_name_list[i],i,gpu_id)

if __name__ == "__main__":
    obj_list_path="../assets/object/DGN_obj/valid_split/bodex_all.json"
    with open(obj_list_path,"r") as f:
        obj_list=json.load(f)
    
    cpu_id_num=6 #A computer can hold 6 processes at most
    obj_list_list=[[] for i in range(cpu_id_num)]
    for i in range(4):
        obj_list_list[i%cpu_id_num].append(obj_list[i])
    gpu_id_list=[i//2+2 for i in range(cpu_id_num)]#Useless Here, Because the code is not GPU-based
    pool = mp.Pool(processes=cpu_id_num+1)
    Res=[pool.apply_async(main, args=(gpu_id_list[i],obj_list_list[i])) for i in range(cpu_id_num)]
    pool.close()
    pool.join()
    