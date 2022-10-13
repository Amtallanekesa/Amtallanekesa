#!/usr/bin/env python
# coding: utf-8

# In[1]:


simple_list=['sharon','jeff','niv','ian']
print(simple_list)


# In[3]:


import pandas as pd
simple_list=['sharon','jeff','niv','ian']
data=pd.DataFrame(simple_list)
print(data)


# In[5]:


import pandas as pd
grocery_list=['onions','garlic','ginger','beans','tomatoes','spinach','tumeric']
price_list=['20','10','50','150','50','25','100']
named_column={'Name':grocery_list,'Price':price_list}

data=pd.DataFrame(named_column)
print(data)


# In[6]:


import pandas as pd
student_list=['Teddy','Chayenne','Chad','Kimberly']
weight_list=['30','32','31','29']
marks_list=['95','91','89','76']
named_column={'Name':student_list,'Weight':weight_list,'Mark':marks_list}
data=pd.DataFrame(named_column)
print(data)


# In[8]:


import pandas as pd
car_list=['mercedes','rangerover','fielder','bmw']
price_list=['2m','5m','1m','10m']
colour_list=['blue','black','grey','white']
seat_numbers=['4','5','5','5']
named_column={'Name':car_list,'Price':price_list,'colour':colour_list,'Seats':seat_numbers}
data=pd.DataFrame(named_column)
print(data)


# In[21]:


#Selecting a column
import pandas as pd
car_list=['mercedes','rangerover','fielder','bmw']
price_list=['2m','5m','1m','10m']
coluor_list=['blue','black','grey','white']
seat_numbers=['4','5','5','5']
named_column={'Name':car_list,'Price':price_list,'colour':colour_list,'Seats':seat_numbers}
data=pd.DataFrame(named_column)
selected_column=data['Seats']
print(selected_column)


# In[22]:


import pandas as pd
car_list=['mercedes','rangerover','fielder','bmw']
price_list=['2m','5m','1m','10m']
colour_list=['blue','black','grey','white']
seat_numbers=['4','5','5','5']
named_column={'Name':car_list,'Price':price_list,'colour':colour_list,'Seats':seat_numbers}
data=pd.DataFrame(named_column)
selected_column=data['colour']
print(selected_column)


# In[26]:


#selecting a value in a certain cell
car_list=['mercedes','rangerover','fielder','bmw']
price_list=['2m','5m','1m','10m']
colour_list=['blue','black','grey','white']
seat_numbers=['4','5','5','5']
named_column={'Name':car_list,'Price':price_list,'colour':colour_list,'Seats':seat_numbers}
data=pd.DataFrame(named_column)
selected_column=data['colour']
selected_row=data.iloc[2]['colour']
print(selected_row)


# In[27]:


#selecting value in certain cell
import pandas as pd
car_list=['mercedes','rangerover','fielder','bmw']
colour_list=['blue','black','grey','white']
seat_numbers=['4','5','5','5']
named_column={'Name':car_list,'colour':colour_list,'Seats':seat_numbers}
data=pd.DataFrame(named_column)
selected_column=data['Seats']
selected_row=data.iloc[0]['Seats']
print(selected_row)


# In[2]:


import pandas as pd
data=pd.read_excel('C:\\Users\\namtala\\Downloads\\Road construction bids.xls')
print(data)


# In[25]:


import pandas as pd
data=pd.read_csv('C:\\Users\\namtala\Downloads\\annual-enterprise-survey-2021-financial-year-provisional-csv.csv')
print(data)


# In[26]:


print(data.info)


# In[27]:


print(data.info())


# In[28]:


print(data.head())


# In[32]:


code=data['Industry_code_ANZSIC06']
print(code)


# In[34]:


name=data['Variable_name']
print(name)


# In[36]:


codes=data.Variable_code
print(codes)


# In[37]:


the_value=data.Value
print(the_value)


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('C:\\Users\\namtala\Downloads\\annual-enterprise-survey-2021-financial-year-provisional-csv.csv')
value_x=data['Industry_name_NZSIOC']
value_y=data['Value']
plt.plot(value_x,value_y)
plt.show()


# In[3]:


import pandas as pd
r=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQk-Kzj1C6wFhfcE7pr6vJzbuUKUHrG7c8R23Tef0pRpySGQ4EdzqpeeYSqLVLzlpNXzkXJkU6Qmjne/pub?gid=250414325&single=true&output=csv')
print(r)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x_values=np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
x_values=pd.Series(x_values)
y_values=np.array([7.38,1.09,2.46,4.1,12.84,1.37,1.09,3.55,7.65,0,3.01,3.28,2.46,7.38,6.83,7.65,0,4.92,4.1,6.28,4.37,1.09,2.46,0,4.64,0])
y_values=pd.Series(y_values)
plt.plot(x_values, y_values)
plt.show()


# In[64]:


plate_str='FRQ****'
def lookup_plate(plate_str,colour=None):
 if type(plate_str) != str:
  print("Error! please input a string")
  return False
 elif len(plate_str) != 7:
  print("Error! Licence plate must have 7 characters.Use an * for missing characters.")
  return False
 elif plate_str == 'FRQ****':
  if color is None:
   print('''
   Fred Frequentist
    John W. Tukey
    Ronald Aylmer Fisher
    Karl Pearson
    Gertrude Cox
    Kirstine Smith
    ''')
   return True
 elif color == 'Green':
  print('''
  Fred Frequentist
  Ronald Aylmer Fisher
  Gertrude Cox
  Kirstine Smith
    ''')
  return True
elif plate_str == 'EXAMPLE':
 print('''
 Christopher Eccleston
 Matt Smith
 David Tenant
 Peter Capaldi
 Jodie Whittaker
 ''')
  return True
lookup_plate(plate, color='Green')


# In[6]:


x= 21
if x % 2 == 0:
    print("is an even number")
elif x % 2 != 0:
    print("is an odd number")
else:
    print("try again")


# In[53]:


x= 3
if x> 0:
     for i in range(2,x):
        if x % 2 == 0:
            print("not prime number")
        else:
            print("prime number")


# In[64]:


x= -2
if x> 0 and x % 2 == 0:
    for i in range (2,x):
        print("not a prime number")
    else:
        print("prime number")
else:
            print("give a positive number")
        


# In[68]:


x=12
y= 10
print ('greater') if x > y else print('equal') if x == y else print('smaller') 


# In[1]:


import pandas as pd
credit_records = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQk-Kzj1C6wFhfcE7pr6vJzbuUKUHrG7c8R23Tef0pRpySGQ4EdzqpeeYSqLVLzlpNXzkXJkU6Qmjne/pub?gid=250414325&single=true&output=csv')
print(credit_records.tail())


# In[5]:


print(credit_records.info())


# In[6]:


freq= credit_records['frequency']
print(freq)


# In[8]:


index= credit_records['letter_index']
print(index)


# In[9]:


let= credit_records.letter
print(let)


# In[10]:


import pandas as pd
credits= pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=1503578122&single=true&output=csv')
print(credits)


# In[12]:


suspects= credits.suspect
print(suspects)


# In[14]:


height_inches= 90
print(height_inches >= 85)


# In[15]:


import pandas as pd
mpr= pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=1929816979&single=true&output=csv')
print(mpr)


# In[17]:


less_than_age_4= mpr[mpr.Age< 4]
print(less_than_age_4)


# In[24]:


Missing_dogs=mpr[mpr.Status=='Still_Missing']
print(Missing_dogs)


# In[29]:


Dog_Breed=mpr[mpr['Dog Breed'] !='Poodle']
print(Dog_Breed)


# In[30]:


Owner_Name= mpr[mpr['Owner Name'] !='Dr. Apache']
print(Owner_Name)


# In[49]:


purchase= credits[credits.location=='Clothing Club']
print(purchase)


# In[58]:


purchase=credits[credits.date!= 43104]
print(purchase)


# In[56]:


if credits.empty == True:
    print("Dataframe is empty")
else:
    print("Dataframe not empty")


# In[61]:


purchase= 'after kidnapping'
print(purchase!='after kidnapping')


# In[63]:


import matplotlib.pyplot as plt
data={'day_of_week': ['M','Tu','w','Th','F'],'hours_worked': ['8','5','3','5','8']}
Deshaun=pd.DataFrame(data)
plt.plot(Deshaun.day_of_week, Deshaun.hours_worked)
plt.show()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data={'day_of_week':['M','Tu','w','Th','F'],'hours_worked':['10','2','1','0','0']}
Aditya=pd.DataFrame(data)
data={'day_of_week':['M','Tu','w','Th','F'],'hours_worked':['0','0','5','9','5']}
mengfei=pd.DataFrame(data)
data={'day_of_week': ['M','Tu','w','Th','F'],'hours_worked': ['8','5','3','5','8']}
Deshaun=pd.DataFrame(data)
plt.plot(Deshaun.day_of_week, Deshaun.hours_worked, label='Deshaun')
plt.plot(Aditya.day_of_week, Aditya.hours_worked, label='Aditya')
plt.plot(mengfei.day_of_week, mengfei.hours_worked, label='mengfei')
plt.legend()
plt.show()


# In[14]:


data={'months':['Jan','Feb','Mar','Apr','Jun'],'hours_worked':[160,185,182,195,50]}
six_months=pd.DataFrame(data)
plt.plot(six_months.months, six_months.hours_worked)
plt.text(2.5,80,"Missing June Data")
plt.show()


# In[18]:


data={'months':['Aug','Sep','Oct','Nov'],'hours_worked':[170,155,140,42]}
four_months=pd.DataFrame(data)
plt.plot(four_months.months, four_months.hours_worked,label='Abel')
plt.text(1.5,100,"On Paternity Leave")
plt.legend()
plt.show()


# In[20]:


data=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=2021138877&single=true&output=csv')
print(data)


# In[8]:


import matplotlib.pyplot as plt
data=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=2021138877&single=true&output=csv')
plt.plot(data["Year"], data["Phoenix Police Dept"], label='Phoenix',marker='o',color="DarkCyan")
plt.plot(data["Year"], data["Los Angeles Police Dept"], label='Los Angeles', linestyle=':', color="red")
plt.plot(data["Year"], data["Philadelphia Police Dept"], label='Philadelphia', marker='s', color="purple")
plt.legend()
plt.show()


# In[34]:


plt.plot(data["Year"], data["New York City Police Dept"], label="New York City",color='yellow',marker='o')
plt.plot(data["Year"], data["Philadelphia Police Dept"], label="Philadelphia", linestyle='-',color='green')
plt.plot(data["Year"], data["Los Angeles Police Dept"], label="Los Angeles",marker='d', color='red')
plt.plot(data["Year"], data["Phoenix Police Dept"], label="Phoenix",marker='d', color='blue')
plt.legend()
plt.show()


# In[44]:


plt.style.use("classic")
plt.plot(data["Year"],data["New York City Police Dept"], label="New York City")
plt.plot(data["Year"],data["Philadelphia Police Dept"],label="Philadelphia")
plt.plot(data["Year"],data["Los Angeles Police Dept"],label="Los Angeles")
plt.plot(data["Year"],data["Phoenix Police Dept"],label="Phoenix")
plt.legend()
plt.show()


# In[38]:


print(plt.style.available)


# In[47]:


print(plt.marker.available)


# In[48]:


data=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=46723742&single=true&output=csv')
print(data)


# In[49]:


plt.plot(data["letter"],data["frequency"],linestyle=':',color='gray',label="Ransom")
plt.legend()
plt.show()


# In[50]:


suspect1=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=1271401594&single=true&output=csv')
print(suspect1)


# In[53]:


plt.plot(suspect1["letter"],suspect1["frequency"],label="Fred Frequentist",marker='d',color='gray')
plt.legend()
plt.show()


# In[54]:


suspect2=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=592723203&single=true&output=csv')
print(suspect2)


# In[55]:


plt.plot(suspect2.letter,suspect2.frequency,label="Gertrude Cox",marker='o',color='gray')
plt.legend()
plt.show()


# In[59]:


plt.plot(data["letter"],data["frequency"],linestyle=':',color='gray',label="Ransom")
plt.plot(suspect1["letter"],suspect1["frequency"],label="Fred Frequentist",marker='d',color='gray')
plt.plot(suspect1["letter"],suspect2["frequency"],label="Gertrude Cox",marker='o',color='gray')
plt.xlabel('letter')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[11]:


cellphone=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=65169422&single=true&output=csv')
print(cellphone.info())


# In[65]:


plt.scatter(cellphone.x,cellphone.y,color='green',marker='s',alpha=0.2)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()


# In[67]:


hours=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=193007150&single=true&output=csv')
print(hours)


# In[69]:


plt.bar(hours["officer"],hours["avg_hours_worked"],color='pink')
plt.show()


# In[70]:


plt.bar(hours["officer"],hours["avg_hours_worked"],yerr=hours["std_hours_worked"],color='green')
plt.show()


# In[71]:


hours=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=902082112&single=true&output=csv')
print(hours)


# In[13]:


hours=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=902082112&single=true&output=csv')
plt.bar(hours["officer"],hours["desk_work"],color='purple',label="Desk Work")
plt.legend()
plt.show()


# In[14]:


plt.bar(hours["officer"],hours["field_work"],color='maroon',label="Field Work")
plt.legend()
plt.show()


# In[81]:


plt.bar(hours["officer"],hours["desk_work"],color='yellow',label="Desk Work")
plt.bar(hours["officer"],hours["field_work"],bottom=hours["desk_work"],label="Field Work")
plt.legend()
plt.show()


# In[16]:


puppies= pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=2075930217&single=true&output=csv')
print(puppies.info())


# In[18]:


plt.hist(puppies.weight)
plt.xlabel('puppy weight(lbs)')
plt.ylabel('number of puppies')
plt.show()


# In[22]:


plt.hist(puppies.weight,bins=60,range=(1,40))
plt.xlabel('puppy weight(lbs)')
plt.ylabel('number of puppies')
plt.show()


# In[26]:


gravel=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQYe0Lsydx6HyVe4FBHjBsLxvAsjii-qobZ-oGGZryFJWRFjemjs43BwZQONRAhLCawSUBkBDI__-Xw/pub?gid=1707010477&single=true&output=csv')
print(gravel.head())


# In[34]:


plt.hist(gravel.radius,bins=40,range=(2,8))
plt.show()


# In[37]:


plt.hist(gravel.radius,bins=50,range=(1,12))
plt.show()


# In[39]:


plt.hist(gravel.radius,bins=40,range=(2,8),density=True)
plt.show()


# In[51]:


import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
plt.hist(gravel.radius,bins=40,range=(2,8),density=True)
plt.xlabel('Gravel radius(mm)')
plt.ylabel('Frequency')
plt.title('Sample from Shoeprint')
plt.show()


# In[52]:


import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)
plt.hist(gravel.radius,bins=40,range=(2,8),density=True)
plt.xlabel('Gravel radius(mm)')
plt.ylabel('Frequency')
plt.title('Sample from Shoeprint')
plt.show()


# In[6]:


basic=input('please enter your basic salary:')
sum_taxes=5000
salary=int(basic)-sum_taxes
if(salary>20000):
    print('You earned',salary,'less than the total',sum_taxes)
elif(salary<10000):
    print('You earned',salary,'less than the total',sum_taxes)
else:
    print('You earned an amount equal to taxes')
            


# In[1]:


names=['sharon','jeff','niv','winnie','teddy']
names_sorted_a=sorted(names,reverse=False)
names_sorted_a


# In[5]:


points_game1=[50,40,60,70,80]
points_game2=[76,81,53,92,67]
diffs=[]
for x,y in zip(points_game1,points_game2):
    diffs.append(abs(x-y))
diffs


# In[6]:


marks_3=[12,17,56,38]
marks_4=[15,23,54,69]
diffs=[]
for x,y in zip(marks_3,marks_4):
    diffs.append(abs(x-y))
diffs


# #table
# |exam|student_ID|grade|
# |:...|:........:|:...:|
# |0  1|         1| 86.0|
# |1  1|         2| 65.0|
# |2  1|         3| 70.0|
# |3  1|         4| 98.0|
# |4  1|         5| 89.0|
# 

# In[22]:


import pandas as pd
grades=pd.read_csv('file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Functions.zip\\Ex_Files_Python_Data_Functions\\Exercise%20Files\\grades.csv')
grades.head()


# In[33]:


def mean_atleast_70(student_id):
    """compute mean grade across all exams for students with given student_id.Treat missing exam grades as zeros. If mean grade is at least 70, return True. Otherwise return False"""
    mean_grade=grades.loc[grades['student_id']==student_id]['grade'].fillna(0).mean()
    return mean_grade>=70


# In[34]:


# test mean_grade on student_id 1
assert mean_atleast_70(1)== False,'test failed'
print('test passed')


# In[35]:


# sequence containing all distinct student ids
student_ids=grades['student_id'].unique()
student_ids


# In[36]:


list(filter(mean_atleast_70,student_ids))


# In[37]:


import numpy as np
array0=np.array([])
array0


# In[38]:


list1=[3,6,9,12]
array1=np.array([list1])
array1


# In[39]:


list2=[[1,2,3],[4,5,6],[7,8,9]]
array2=np.array([list2])
array2


# In[50]:


#create a numpy array containing all even integers from 1 to 11 including 1 and 11
np.arange(1,12,2)


# In[52]:


np.zeros((4,5))


# In[53]:


np.ones((3,4))


# In[54]:


np.linspace(2,3,9)


# In[58]:


np.random.randint(20,50,10)


# In[61]:


array2.shape


# In[63]:


array2.reshape(9)


# In[3]:


import numpy as np
list2=[[1,2,3],[4,5,6],[7,8,9]]
array2=np.array([list2])
np.median(array2)


# In[4]:


np.var(array2)


# In[5]:


np.exp(array2)


# In[6]:


np.square(array2)


# In[7]:


np.log(array2)


# In[9]:


np.sum(array2,axis=1)


# In[26]:


from scipy import linalg
list3=[[8,6],[3,4]]
array3=np.array([list3])
array3


# In[28]:


linalg.det(array3)


# In[29]:


from scipy import stats
B=stats.binom(10,0.4)
B


# In[30]:


#compute probability mass function at 2
B.pmf(2)


# In[31]:


#compute value of its cumulative density function at 3
B.cdf(3)


# In[32]:


#declare P to be a poisson discrete random variable with parameter 2
P=stats.poisson(2)


# In[34]:


#compute value of its probability mass function at 2
P.pmf(2)


# In[35]:


#declare G to be a geometric discrete random variable with parameter 0.25
G=stats.geom(0.25)


# In[36]:


G.pmf(3)


# In[37]:


N=stats.norm(0,1)


# In[38]:


N.pdf(0.1)


# In[39]:


E=stats.expon(4)


# In[40]:


X=stats.beta(1,5)


# In[51]:


import pandas as pd
pd.Series(np.random.randint(-10,11,size=8),index=['a','b','c','d','e','f','g','h'])


# In[53]:


pd.Series(150,index=np.arange(1,11))


# In[56]:


points={'player1':pd.Series([15,10,20,25],['game1','game2','game3','game4']),'player2':pd.Series([10,15,23,27],['game1','game2','game3','game4'])}
pd.DataFrame(points)


# In[57]:


#create a dictionary calles sales
sales={'foodtruck1':[120,150,133,128],'foodtruck2':[155,147,196,122]}
sales


# In[58]:


#create a python DataFrame from the dictionary above
pd.DataFrame(sales,index=['day1','day2','day3','day4'])


# In[59]:


pd.DataFrame(sales)


# In[70]:


import pandas as pd
grades=pd.read_csv('file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Functions.zip\\Ex_Files_Python_Data_Functions\\Exercise%20Files\\grades.csv')
grades.loc[:,['student_id','grade']]


# In[76]:


grades.iloc[0]


# In[77]:


grades.iloc[[0,10],:]


# In[81]:


grades.iloc[:,[1,2]]


# In[82]:


grades.iloc[[34,35],[0,2]]


# In[5]:


import pandas as pd
grades=pd.read_csv('file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Functions.zip\\Ex_Files_Python_Data_Functions\\Exercise%20Files\\grades.csv')
grades['grade']=grades['grade'].fillna(0)
grades


# In[15]:


import pandas as pd
grades=pd.read_csv('file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Functions.zip\\Ex_Files_Python_Data_Functions\\Exercise%20Files\\grades.csv')
grades=grades.drop(columns=['student_id'],axis=1)
grades


# In[18]:


points={'player1':pd.Series([15,10,20,25],['game1','game2','game3','game4']),'player2':pd.Series([10,15,23,27],['game1','game2','game3','game4'])}
pd.DataFrame(points)
pd.merge(grades,points,on='SID')


# In[21]:


import pandas as pd
grades=pd.read_csv('file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Functions.zip\\Ex_Files_Python_Data_Functions\\Exercise%20Files\\grades.csv')
grades.drop(columns='exam').groupby('student_id').mean()


# In[25]:


import pandas as pd
grades=pd.read_csv('file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Functions.zip\\Ex_Files_Python_Data_Functions\\Exercise%20Files\\grades.csv')
grades.drop(columns='student_id').groupby('exam').median()


# In[29]:


import matplotlib.pyplot as plt
plt.scatter(grades['grade'],grades['student_id'])
plt.xlabel('student_id')
plt.ylabel('grade')
plt.show()


# In[33]:


import seaborn as sns
sns.boxplot(y='grade',data=grades);


# In[34]:


sns.boxplot(x='student_id',data=grades);


# In[35]:


sns.boxplot(x='grade',y='student_id',data=grades);


# In[71]:


sns.kdeplot(grades['grade'],grades['student_id']);


# In[62]:


import pandas as pd
import seaborn as sns
bids=pd.read_excel('C:\\Users\\namtala\\Downloads\\Road construction bids.xls')
bids


# In[75]:


sns.kdeplot(bids['cost'],bids['daysest'],shade=True);


# In[74]:


sns.violinplot(y='btpratio',data=bids);


# In[77]:


sns.violinplot(x='b2b1rat',y='b3b1rat',data=bids);


# In[78]:


sns.heatmap(bids);


# In[1]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[3]:


series_obj=Series(np.arange(8),index=['row1','row2','row3','row4','row5','row6','row7','row8'])
series_obj


# In[4]:


series_obj['row3']


# In[5]:


series_obj[[0,1]]


# In[6]:


np.random.seed(25)
DF_obj=DataFrame(np.random.rand(36).reshape(6,6),index=['row1','row2','row3','row4','row5','row6'],columns=['column1','column2','column3','column4','column5','column6'])
DF_obj


# In[8]:


DF_obj.loc[['row3','row5'],['column1','column2']]


# In[6]:


import numpy as np
from pandas import DataFrame
np.random.seed(25)
DF_obj=DataFrame(np.random.rand(36).reshape(6,6),index=['row1','row2','row3','row4','row5','row6'],columns=['column1','column2','column3','column4','column5','column6'])
DF_obj['row3':'row7']


# In[7]:


DF_obj<.2


# In[9]:


DF_obj[DF_obj>6]


# In[12]:


DF_obj['row1','row2','row3']=8
DF_obj


# In[15]:


import pandas as pd
from pandas import Series
series_obj=Series(np.arange(8),index=['row1','row2','row3','row4','row5','row6','row7','row8'])
series_obj['row1','row2','row3']=9
series_obj


# In[19]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
series_obj=Series(['row1','row2','row3',missing,'row5','row6','row7',missing])
series_obj


# In[20]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
series_obj=Series(['row1','row2','row3',missing,'row5','row6','row7',missing])
series_obj


# In[21]:


series_obj.isnull()


# In[23]:


np.random.seed(25)
DF_obj=DataFrame(np.random.rand(36).reshape(6,6))
DF_obj


# In[24]:


DF_obj.loc[3:5,0]=missing
DF_obj.loc[1:4,5]=missing
DF_obj


# In[25]:


filled_DF=DF_obj.fillna(0)
filled_DF


# In[26]:


filled_DF=DF_obj.fillna({0:0.1,5:1.25})
filled_DF


# In[27]:


filled_DF=DF_obj.fillna(method='ffill')
filled_DF


# In[28]:


np.random.seed(25)
DF_obj=DataFrame(np.random.rand(36).reshape(6,6))
DF_obj.loc[3:5,0]=missing
DF_obj.loc[1:4,5]=missing
DF_obj


# In[29]:


DF_obj.isnull().sum()


# In[31]:


DF_no_nan=DF_obj.dropna()
DF_no_nan


# In[35]:


DF_no_nan=DF_obj.dropna(axis=1)
DF_no_nan


# In[36]:


DF_obj=DataFrame({'column1':[1,1,2,2,3,3,3],'column2':['a','a','b','b','c','c','c'],'column3':['A','A','B','B','C','C','C']})
DF_obj


# In[37]:


DF_obj.duplicated()


# In[38]:


DF_obj.drop_duplicates()


# In[39]:


DF_obj=DataFrame({'column1':[1,1,2,2,3,3,3],'column2':['a','a','b','b','c','c','c'],'column3':['A','A','B','B','C','D','C']})
DF_obj


# In[40]:


DF_obj.drop_duplicates(['column3'])


# In[41]:


DF_obj=pd.DataFrame(np.arange(36).reshape(6,6))
DF_obj


# In[42]:


DF_obj_2=pd.DataFrame(np.arange(15).reshape(5,3))
DF_obj_2


# In[44]:


pd.concat([DF_obj,DF_obj_2],axis=1)


# In[45]:


DF_obj.drop([2,3])


# In[47]:


DF_obj.drop([2,3],axis=1)


# In[48]:


series_obj=pd.Series(np.arange(6))
series_obj.name='added variable'
series_obj


# In[51]:


variable_added=DataFrame.join(DF_obj,series_obj)
variable_added


# In[52]:


added_datatable=variable_added.append(variable_added,ignore_index=False)
added_datatable


# In[53]:


added_datatable=variable_added.append(variable_added,ignore_index=True)
added_datatable


# In[54]:


DF_sorted=DF_obj.sort_values(by=(5),ascending=[False])
DF_sorted


# In[55]:


address='file:///C:\\Users\\namtala\\AppData\\Local\\Temp\\Temp1_Ex_Files_Python_Data_Science_EssT_Pt_1.zip\\Ex_Files_Python_Data_Science_EssT_Pt_1\\Exercise%20Files\\Data\\mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
cars


# In[57]:


cars_groups=cars.groupby(cars['cyl'])
cars_groups.mean()


# In[58]:


cars_groups=cars.groupby(cars['am'])
cars_groups.mean()


# In[4]:


import pandas as pd
import numpy as np
address='file:///C:/Users/namtala/AppData/Local/Temp/Temp1_Ex_Files_Python_Data_Science_EssT_Pt_1.zip/Ex_Files_Python_Data_Science_EssT_Pt_1/Exercise%20Files/Data/mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
cars.head()   


# In[6]:


cars_groups=cars.groupby(cars['qsec'])
cars_groups.mean()


# In[7]:


import pandas as pd
import numpy as np
from numpy.random import randn
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from matplotlib import rcParams


# In[9]:


x=range(1,10)
y=[1,2,3,4,0,4,3,2,1]
plt.plot(x,y)


# In[10]:


address='file:///C:/Users/namtala/AppData/Local/Temp/Temp1_Ex_Files_Python_Data_Science_EssT_Pt_1.zip/Ex_Files_Python_Data_Science_EssT_Pt_1/Exercise%20Files/Data/mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
mpg=cars['mpg']


# In[11]:


mpg.plot()


# In[14]:


df=cars[['disp','qsec','mpg']]
df.plot()


# In[15]:


plt.bar(x,y)


# In[20]:


mpg.plot(kind='bar')


# In[24]:


mpg.plot(kind='barh')


# In[25]:


x=[1,2,3,4,0.5]
plt.pie(x)
plt.show()


# In[26]:


plt.pie(x)
plt.savefig('pie_chart.png')
plt.show()


# In[27]:


get_ipython().run_line_magic('pwd', '')


# In[4]:


from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']= 5,4


# In[9]:


import matplotlib.pyplot as plt
x=range(1,10)
y=[1,2,3,4,0,4,3,2,1]
fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])
ax.plot(x,y)


# In[10]:


fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])
ax.set_xlim([1,9])
ax.set_ylim([0,5])
ax.set_xticks([0,1,2,4,5,6,8,9,10])
ax.set_yticks([0,1,2,3,4,5])
ax.plot(x,y)


# In[15]:


fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])
ax.set_xlim([1,9])
ax.set_ylim([0,5])
ax.set_xticks([0,1,2,4,5,6,8,9,10])
ax.set_yticks([0,1,2,3,4,5])
ax.grid()
ax.plot(x,y)


# In[16]:


fig=plt.figure()
fig, (ax1,ax2)=plt.subplots(1,2)
ax1.plot(x)
ax2.plot(y)


# In[17]:


x=range(1,10)
y=[1,2,3,4,0.5,4,3,2,1]
plt.bar(x,y)


# In[23]:


wide=[.5,.5,.5,.7,.7,.7,.5,.5,.5]
color=['salmon']
plt.bar(x,y,width=wide,color=color,align='center')


# In[26]:


x=range(1,10)
y=[1,2,3,4,0.5,4,3,2,1]
plt.bar(x,y,width=[.5,.5,.5,.7,.7,.7,.5,.5,.5],color=['salmon'],align='edge')


# In[32]:


import pandas as pd
address='file:///C:/Users/namtala/AppData/Local/Temp/Temp1_Ex_Files_Python_Data_Science_EssT_Pt_1.zip/Ex_Files_Python_Data_Science_EssT_Pt_1/Exercise%20Files/Data/mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['cars_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
df=cars[['cyl','mpg','wt']]
color_theme='darkgray','lightsalmon','powderblue'
df.plot(color=color_theme)


# In[33]:


z=[1,2,3,4,.5]
plt.pie(z)
plt.show()


# In[38]:


color_theme=['#A9A9A9','#FFA07A','#B0E0E6','#FFE4C4','#BDB76B']
plt.pie(z,colors=color_theme)
plt.show()


# In[41]:


x1=range(0,10)
y1=[10,9,8,7,6,5,4,3,2,1]
plt.plot(x,y)
plt.plot(x1,y1)


# In[42]:


plt.plot(x,y,ds='steps',lw=5)
plt.plot(x1,y1,ls='--',lw=10)


# In[43]:


plt.plot(x,y,marker='1',mew=20)
plt.plot(x1,y1,marker='+',mew=15)


# In[ ]:




