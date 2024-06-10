#!/usr/bin/env python
# coding: utf-8

# In[7]:


#List
li = [1, 2, 3, 4, 5]
print("Original list: ",li)
#Basic operatins on list.
li.append(6) #Adding element to list
li.remove(5) #Removing element from list
li[2] = 10   #Modifying element in list
print("Modified list: ",li)

#Dictionary
di = {'Name':"John", 'Age':19, 'Gender':'M'}
print("Original dictionary: ",di)
#Basic operatins on dictionary.
di['City'] = "New York" #Adding element to dictionary
del di["Gender"]        #Removing element from dictionary
di["Name"] = "Jack"     #Modifying element in dictionary
print("Modified dictionary: ",di)

#Set
s = {1, 2, 3, 1, 2, 4, 5, 5}
print("Original set: ",s)
#Basic operatins on set.
s.add(6)     #Adding element to set
s.discard(5) #Removing element from set
#Set items are unchangable so we can only add and remove elements but we cannot modify them.
print("Modified set: ",s)

