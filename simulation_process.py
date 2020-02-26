import numpy as np
from new_model import get_bayes_model,get_svr_model,get_arima_forecast,get_lstm_model,train_lstm,lstm_predict

def PM_workload_prediction(prediction_model,lag,Task_data, PM_VM_list, current_time,PM_num):

    model=prediction_model
    PM_workload_data=[]
    PM_workload=[]
    for i in range(PM_num):
        Task_list=PM_VM_list[i]
        next_Task_data=[]
        PMload=0
        for j in range (len(Task_list)):
            seris_data=Task_data[:,Task_list[j]]
            x_train=np.zeros(lag)
            for k in range(lag):
                if current_time-k<0:
                    x_train[lag-1-k]=0
                else:
                    x_train[lag-1-k]=seris_data[current_time-k]
            x_train= x_train.reshape(1,lag,1)
            predict_y=lstm_predict(x_train,model)
            
            y=predict_y[0][0]*0.4+0.6*seris_data[current_time+1]#predict_y[0][0]
            PMload+=y
            next_Task_data.append(y)
        PM_workload_data.append(next_Task_data)
        PM_workload.append(PMload)
    return PM_workload_data, PM_workload

def PM_current_workload(Task_data, PM_VM_list, current_time,PM_num):
    PM_workload_data=[]
    PM_workload=[]
    for i in range(PM_num):
        Task_list=PM_VM_list[i]
        PMload=0
        current_task_workload=[]
        for j in range (len(Task_list)):
            current_task_data=Task_data[current_time][Task_list[j]]
            PMload+=current_task_data
            current_task_workload.append(current_task_data)
        PM_workload_data.append(current_task_workload)
        PM_workload.append(PMload)
    return PM_workload_data, PM_workload
        
def find_task_migration(PM_migration_index,target,PM_next_workload_data):
    Migration_PM_workload=PM_next_workload_data[PM_migration_index]
    index=0;
    min_margin=10
    for i in range(len(Migration_PM_workload)):
        margin=abs(Migration_PM_workload[i]-target)
        if margin<min_margin:
            index=i
            min_margin=margin
    #index=Migration_PM_workload.index(max(Migration_PM_workload))
    task_load=Migration_PM_workload[index]
    task_index=index
    return task_index, task_load
    
def migration_process(PM_migration_index,migration_task_index,PM_acception_index,PM_vm_list,PM_next_workload_data,PM_next_workload):
    migration_task=PM_vm_list[PM_migration_index][migration_task_index]
    PM_vm_list[PM_acception_index].append(migration_task)
    del PM_vm_list[PM_migration_index][migration_task_index]
    migration_task_load=PM_next_workload_data[PM_migration_index][migration_task_index]
    del PM_next_workload_data[PM_migration_index][migration_task_index]
    PM_next_workload_data[PM_acception_index].append(migration_task_load)
    PM_next_workload[PM_acception_index]+=migration_task_load
    PM_next_workload[PM_migration_index]-=migration_task_load

def observation_state(PM_workload,PM_status):
    if PM_status==1:
        Threshold1=0.9
        Threshold2=0.4
    if PM_status==0:
        Threshold1=0.2
        Threshold2=0.1
    PM_state=np.zeros(len(PM_workload))
    for i in range(len(PM_workload)):
        if PM_workload[i]>= Threshold1:
            PM_state[i]=3
        elif PM_workload[i]>= Threshold2:
            PM_state[i]=2
        else:
            PM_state[i]=1
   # PM_state.reshape((1,len(PM_workload)))
    
    return PM_state
    
def env_step(PM_index,PM_current_workload_data,PM_current_workload,PM_vm_list,action,PM_VM_list_status):
    SLo_before_action=len(list(filter(lambda x: x>0.95 , PM_current_workload)))
    initial_state=observation_state(PM_current_workload,1)
    ### Action_process
    reward=0
    if action==297:
       state=observation_state(PM_current_workload,1)
       if state[PM_index]==1 or state[PM_index]==2:
            reward+=4
       if state[PM_index]==3:
            reward=-2
       return state, reward,SLo_before_action
    PM_migration=PM_index
    PM_destination=int(action//3)
    if PM_destination>=PM_migration:
        PM_destination+=1
    VM_level=action-(action//3*3)+1
    PM_VM_state=list(observation_state(PM_current_workload_data[PM_migration],0))
    if len(PM_VM_state)==0:
        reward=0
        state=observation_state(PM_current_workload,1)
        print("NO VM")
        return state, reward,SLo_before_action
    temp_list=[]
    for i,j in enumerate(PM_VM_state):
        if j==VM_level:
            temp_list.append((i,j))
            
    if len(temp_list)!=0:   
        index=np.random.randint(len(temp_list))
        VM_index, VM_state=temp_list[index]
    else:
        index=np.random.randint(len(PM_VM_state))
        VM_index=index
        VM_state=PM_VM_state[VM_index]
    #print(PM_VM_state[PM_migration])
    #print("migration ",PM_migration,PM_destination,VM_state)
    PM_after_workload=PM_current_workload.copy()
    PM_after_workload[PM_migration]=PM_current_workload[PM_migration]-PM_current_workload_data[PM_migration][VM_index]
    PM_after_workload[PM_destination]=PM_current_workload[PM_destination]+PM_current_workload_data[PM_migration][VM_index]
    #PM_vm_list[PM_destination].append(PM_vm_list[PM_migration][VM_index])
    #del PM_vm_list[PM_migration][VM_index]
    #PM_current_workload_data[PM_destination].append(PM_current_workload_data[PM_migration][VM_index])
    #del PM_current_workload_data[PM_migration][VM_index]
    #SLo_after_action=len(list(filter(lambda x: x>0.95 , PM_after_workload)))
    #reward=SLo_before_action-SLo_after_action
    state=observation_state(PM_after_workload,1)
    
    # if initial_state[PM_destination]==3:
        # reward-=VM_level
        
    # if initial_state[PM_destination]==1 or initial_state[PM_destination]==2:
        # reward+=VM_level

        
    # if initial_state[PM_destination]==2:
         # if state[PM_destination]==3:
             # reward=-2
         # if state[PM_destination]==2:
             # reward+=VM_level
    
    # if initial_state[PM_destination]==1:
         # if state[PM_destination]==3:
             # reward=-2
         # if state[PM_destination]==2:
             # reward+=VM_level
  
    PM_current_workload[PM_migration]-=PM_current_workload_data[PM_migration][VM_index]
    PM_current_workload[PM_destination]+=PM_current_workload_data[PM_migration][VM_index]
    PM_current_workload_data[PM_destination].append(PM_current_workload_data[PM_migration][VM_index])
    del PM_current_workload_data[PM_migration][VM_index]
    if PM_VM_list_status==1:
        PM_vm_list[PM_destination].append(PM_vm_list[PM_migration][VM_index])
        del PM_vm_list[PM_migration][VM_index]
    SLo_after_action=len(list(filter(lambda x: x>0.95 , PM_current_workload)))
    reward=SLo_before_action-SLo_after_action
    return state,reward,SLo_after_action
    







def  get_nn_state(state,i):
    nn_state=[]
    nn_state.append(state[i])
    for j in range(len(state)):
        if j==i:
            continue
        else:
            nn_state.append(state[j])
    
    m_state=np.array(nn_state)
    
    return m_state
            
    
    