
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import re


# In[18]:


#对drector清洗：
def sub_name(str1):
    first_name = []
    a = str1.split(',')
    for i in a:            
        i = re.sub('[ ]','',i)
        if i[0].encode( 'UTF-8' ).isalpha():
            first_name.append(i)
        else:
            b = re.sub('[ A-Za-z?-í-""]','',i)
            first_name.append(b)
    if len(first_name) == len(a):
        str1 = str()
        for i in first_name:
            if len(str1) == 0:
                str1 = str1 +i
            else:
                str1 = str1 + ',' + i
        return str1           


# In[19]:


def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


# In[44]:


list1 = []
if len(list1) == 0:
    print("ok")
else:
    print("No")


# In[48]:


#统计相关的属性
def statistics(attribute,train_data,test_data):
    #增加关于score评分的特征
    num = 0
    count = 0
    z_count = []
    s_count = []
    max_score = []#最高评分
    min_score = []#最低评分
    ave_score = []#平均评分
    max_score_count = []#最大评分对应的票房
    min_score_count = []#最低评分对应的票房

    max_count = []
    min_count = []
    ave_count = []

    num_move = []
    # index_count = []
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []

    G = []
    H = []
    I = []
    # J = []

    for i in test_data[attribute][:]:
        a = i.split(',')
#         print(a)
        for b in a:
            for j in train_data.index:
                if b in train_data[attribute][j].split(','):
                    z_count.append(train_data.score[j])#评分
                    s_count.append(train_data.account[j])
    #                 s_count.append(train_data1.count[j])#票房
    #                 index_count.append(j)
    #                 num = num + train_data.account[j]
                    count = count+1
#             print(z_count)
#             print(s_count)
#             print(count)
            if len(z_count) == 0:
                continue
            else:
                A.append(max(z_count))#最大评分
                B.append(min(z_count))#最小评分
                C.append(sum(z_count)/count)#平均评分
    #             C.append(np.median(z_count))
                D.append(count)
                E.append(s_count[z_count.index(max(z_count))])
                F.append(s_count[z_count.index(min(z_count))])

                G.append(max(s_count))#最大票房
                H.append(min(s_count))#最小票房
                I.append(sum(s_count)/count)#平均票房

                s_count = []
                z_count = []
                count = 0  
        #评分情况
        max_score.append(sum(A)/len(A))#这边不应该取最大值而应该取平均值，即导演的平均值(也可以尝试用最大这里选取平均)
        min_score.append(sum(B)/len(B))
        ave_score.append(sum(C)/len(C))
#         ave_score.append(np.median(z_count))
        num_move.append(sum(D)/len(D))
        max_score_count.append(sum(E)/len(E))
        min_score_count.append(sum(F)/len(F))
        #票房情况
        max_count.append(sum(G)/len(G))
        min_count.append(sum(H)/len(H))
        ave_count.append(sum(I)/len(I))
    #     num_move.append(max(D))

        A = []
        B = []
        C = []
        D = []
        E = []
        F = []  
        G = []
        H = []
        I = []
#     print(max_count)
#     print(min_count)
#     print(ave_count)
#     print(num_move)
#     print(max_score)

#     print(min_score)
#     print(ave_score)
#     print(max_score_count)
#     print(min_score_count)
    return num_move,max_score,min_score,ave_score


# In[25]:


# train_data = pd.read_csv('train_data.csv',encoding='gbk')
# test_data = pd.read_csv('test_data.csv',encoding='gbk')
# train_data = train_data.dropna(axis=0,how='any')

# drector_num = test_data.apply(lambda x:sub_name(x['drector']),axis=1)
# writer_num = test_data.apply(lambda x:sub_name(x['writer']),axis=1)
# actor_num = test_data.apply(lambda x:sub_name(x['actor']),axis=1)
# types_num = test_data.apply(lambda x:sub_name(x['types']),axis=1)
# flim_version = list(map(lambda x: len(test_data.times[x].split('  ')),range(len(test_data))))
# country_number = list(map(lambda x: len(test_data.country[x].split(' ')),range(len(test_data))))
# test_data['types_name'] = pd.DataFrame(types_num)
# test_data['drector_name'] = pd.DataFrame(drector_num)
# test_data['writer_name'] = pd.DataFrame(writer_num)
# test_data['actor_name'] = pd.DataFrame(actor_num)
# test_data['flim_version'] = pd.DataFrame(flim_version)
# test_data['country_number'] = pd.DataFrame(country_number)


# In[26]:


# test_data


# In[53]:


# test_data.writer_name


# In[28]:


# object_name = ['drector_name','writer_name','actor_name','types_name']
# for i in object_name:
#     A,B,C,D= statistics(i,train_data,test_data)
#     test_data[i[0]+'_num_move'] = pd.DataFrame(A)
#     test_data[i[0]+'_max_score'] = pd.DataFrame(B)
#     test_data[i[0]+'_min_score'] = pd.DataFrame(C)
#     test_data[i[0]+'_ave_score'] = pd.DataFrame(D) 


# In[52]:


# z_count = []
# s_count = []
# count = 0
# for i in test_data['writer_name'][:]:
#     a = i.split(',')
#     print(a)
#     for b in a:
#         for j in train_data.index:
#             if b in train_data['writer_name'][j].split(','):
#                 z_count.append(train_data.score[j])#评分
#                 s_count.append(train_data.account[j])
# #                 s_count.append(train_data1.count[j])#票房
# #                 index_count.append(j)
# #                 num = num + train_data.account[j]
#                 count = count+1
#         if len(z_count) == 0:
#             continue
#         else:
#             print(max(z_count))
# #         print(min(z_count))
# #         print(s_count)
#             print(count)


# In[49]:


def change_data():
    train_data = pd.read_csv('train_data.csv',encoding='gbk')
    test_data = pd.read_csv('test_data.csv',encoding='gbk')
    train_data = train_data.dropna(axis=0,how='any')
    
    drector_num = test_data.apply(lambda x:sub_name(x['drector']),axis=1)
    writer_num = test_data.apply(lambda x:sub_name(x['writer']),axis=1)
    actor_num = test_data.apply(lambda x:sub_name(x['actor']),axis=1)
    types_num = test_data.apply(lambda x:sub_name(x['types']),axis=1)
    flim_version = list(map(lambda x: len(test_data.times[x].split('  ')),range(len(test_data))))
    country_number = list(map(lambda x: len(test_data.country[x].split(' ')),range(len(test_data))))
    test_data['types_name'] = pd.DataFrame(types_num)
    test_data['drector_name'] = pd.DataFrame(drector_num)
    test_data['writer_name'] = pd.DataFrame(writer_num)
    test_data['actor_name'] = pd.DataFrame(actor_num)
    test_data['flim_version'] = pd.DataFrame(flim_version)
    test_data['country_number'] = pd.DataFrame(country_number)
    
    object_name = ['drector_name','writer_name','actor_name','types_name']
    for i in object_name:
        A,B,C,D= statistics(i,train_data,test_data)
        test_data[i[0]+'_num_move'] = pd.DataFrame(A)
        test_data[i[0]+'_max_score'] = pd.DataFrame(B)
        test_data[i[0]+'_min_score'] = pd.DataFrame(C)
        test_data[i[0]+'_ave_score'] = pd.DataFrame(D) 
    
        #取导演、演员、编剧的平均创作
    temp1 = list(map(lambda x:(test_data['d_num_move'][x]+test_data['w_num_move'][x])/2,test_data.index))
    temp2 = list(map(lambda x:(test_data['d_num_move'][x]+test_data['a_num_move'][x])/2,test_data.index))
    temp3 = list(map(lambda x:(test_data['w_num_move'][x]+test_data['a_num_move'][x])/2,test_data.index))
    temp4 = list(map(lambda x:(test_data['d_num_move'][x]+test_data['w_num_move'][x]+test_data['a_num_move'][x])/3,test_data.index))
    #取导演、演员、编剧的平均得分
    temp9 = list(map(lambda x:(test_data['d_ave_score'][x]+test_data['w_ave_score'][x])/2,test_data.index))
    temp10 = list(map(lambda x:(test_data['d_ave_score'][x]+test_data['a_ave_score'][x])/2,test_data.index))
    temp11 = list(map(lambda x:(test_data['w_ave_score'][x]+test_data['a_ave_score'][x])/2,test_data.index))
    temp12 = list(map(lambda x:(test_data['d_ave_score'][x]+test_data['w_ave_score'][x]+test_data['a_ave_score'][x])/3,test_data.index))

    test_data['ave_num_move1'] = pd.DataFrame(temp1,index=test_data.index)
    test_data['ave_num_move2'] = pd.DataFrame(temp2,index=test_data.index)
    test_data['ave_num_move3'] = pd.DataFrame(temp3,index=test_data.index)
    test_data['ave_num_move4'] = pd.DataFrame(temp4,index=test_data.index)
    test_data['ave_score1'] = pd.DataFrame(temp9,index=test_data.index)
    test_data['ave_score2'] = pd.DataFrame(temp10,index=test_data.index)
    test_data['ave_score3'] = pd.DataFrame(temp11,index=test_data.index)
    test_data['ave_score4'] = pd.DataFrame(temp12,index=test_data.index)
    
    test_types = list(test_data.types)
    key_s = ['剧情', '动作', '犯罪', '冒险', '科幻', '惊悚', '奇幻', '悬疑', '喜剧', '战争', '动画', '传记', '历史', '西部', '爱情', '灾难', '武侠', '古装', '音乐', '运动', '家庭', '恐怖', '鬼怪', '歌舞', '情色', '儿童', '同性', '悬念', '黑色电影', 'Adult', 'Reality-TV']
    _types = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(0,len(list(test_types))):
        temp = re.sub('[ ]','',test_types[i])
        for j in range(len(_types)):          
            _types[j].append(list(np.where((key_s[j] in temp),[1],[0]))[0])
            if i == len(list(test_types))-1:
                test_data[key_s[j]] = pd.DataFrame(_types[j],index=test_data.index) 
    X_test = test_data.drop(['title','types','drector','writer','actor','times','country','score','types_name','drector_name','writer_name','actor_name','t_num_move'],axis=1)         
    return X_test


# In[50]:


test = change_data()


# In[51]:


test.to_csv('test1_data.csv',index=False,encoding='gbk')


# In[ ]:


# def get_median(data):
#     data.sort()
#     half = len(data) // 2
#     return (data[half] + data[~half]) / 2


# In[ ]:


# data = [7.2,8.7,7.2]
# # 

# np.median(data)


# In[ ]:


# change_data()

