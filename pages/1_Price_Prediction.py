import streamlit as st
import pandas as pd
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

#Page Heading
st.header("Laptop Price Prediction:")

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

DATA_PATH = os.path.join(dir_of_interest, "data", "laptop_price.csv")

df=pd.read_csv(DATA_PATH)
data=df.copy()

#Accepting the required features from user
col1, col2, col3= st.columns(3)
with col1:
    brand = st.selectbox(
        'Select Brand',
        (df.Brand.unique()))
    
with col3:
    operating_system=st.selectbox(
        'Select Operating System',
        (df['OS'].unique()))


with col2:
    processor=st.selectbox(
        'Select Processor Type',
        (df['Processor'].unique()))


col1, col2= st.columns(2)
with col1:
    ram_type=st.selectbox(
        'Select RAM Type',
        (df['ramType'].unique()))


with col2:
    ram_size=st.selectbox(
        'Select RAM Size',
        (df['ramSize'].unique()))


col1, col2= st.columns(2)
with col1:
    disk_type=st.selectbox(
        'Select disk Type',
        (df['diskType'].unique()))

with col2:
    disk_size=st.selectbox(
        'Select disk Size',
        (df['diskSize'].unique()))

#Create dataframe using all these values
sample=pd.DataFrame({"Brand":[brand],"OS":[operating_system], "Processor":[processor],
                   "ramType":[ram_type], "ramSize":[ram_size],
                   "diskType":[disk_type], "diskSize":[disk_size]})

#Convert these values to suitable integer form
#Function to change brand to number
def replace_brand(brand):
    if brand=='Lenovo':
        return 1
    elif brand=='ASUS':
        return 2
    elif brand=='HP':
        return 3
    elif brand=='DELL':
        return 4
    elif brand=='RedmiBook':
        return 5
    elif brand=='realme':
        return 6
    elif brand=='acer':
        return 7
    elif brand=='MSI':
        return 8
    elif brand=='APPLE':
        return 9
    elif brand=='Infinix':
        return 10
    elif brand=='SAMSUNG':
        return 11
    elif brand=='Ultimus':
        return 12
    elif brand=='Vaio':
        return 13
    elif brand=='GIGABYTE':
        return 14
    elif brand=='Nokia':
        return 15
    elif brand=='ALIENWARE':
        return 16    
data['Brand']=data['Brand'].apply(replace_brand)

#Function to change processor to number
def replace_processor(Processor):
    if Processor=='Intel':
        return 1
    elif Processor=='AMD':
        return 2
    elif Processor=='Apple':
        return 3
    elif Processor=='Qualcomm':
        return 4
data['Processor']=data['Processor'].apply(replace_processor)

#Function to change os to number
def replace_os(os):
    if os=='Windows 11':
        return 1
    elif os=='Windows 10':
        return 2
    elif os=='Mac':
        return 3
    elif os=='Chrome':
        return 4
    elif os=='DOS':
        return 5
data['OS']=data['OS'].apply(replace_os)

#Function to change ram type to number
def replace_ram_type(ram_type):
    if ram_type=='DDR4':
        return 1
    elif ram_type=='DDR5':
        return 2
    elif ram_type=='LPDDR4':
        return 3
    elif ram_type=='Unified':
        return 4
    elif ram_type=='LPDDR4X':
        return 5
    elif ram_type=='LPDDR5':
        return 6
    elif ram_type=='LPDDR3':
        return 7      
data['ramType']=data['ramType'].apply(replace_ram_type)

#Function to change ram size to number
def replace_ram_size(ram_size):
    if ram_size=='8GB':
        return 1
    elif ram_size=='16GB':
        return 2
    elif ram_size=='4GB':
        return 3
    elif ram_size=='32GB':
        return 4
data['ramSize']=data['ramSize'].apply(replace_ram_size)

#Function to disk type to number
def replace_disk_type(disk_type):
    if disk_type=='SSD':
        return 1
    elif disk_type=='HDD':
        return 2
    elif disk_type=='EMMC':
        return 3
data['diskType']=data['diskType'].apply(replace_disk_type)

#Function to change disk size to number
def replace_disk_size(disk_size):
    if disk_size=='256GB':
        return 1
    elif disk_size=='512GB':
        return 2
    elif disk_size=='1TB':
        return 3
    elif disk_size=='128GB':
        return 4
    elif disk_size=='64GB':
        return 5
    elif disk_size=='32GB':
        return 6
    elif disk_size=='2TB':
        return 7
data['diskSize']=data['diskSize'].apply(replace_disk_size)

#Split data into X and y
X=data.drop('MRP', axis=1).values
y=data['MRP'].values

#Standarizing the features
std=StandardScaler()
std_fit=std.fit(X)
X=std_fit.transform(X)

#Train the model
xgb=XGBRegressor(learning_rate=0.15, n_estimators=50, max_leaves=0, random_state=42)
xgb.fit(X,y)

#Convert User input to suitable integer form
sample['Brand']=sample['Brand'].apply(replace_brand)
sample['OS']=sample['OS'].apply(replace_os)
sample['Processor']=sample['Processor'].apply(replace_processor)
sample['ramType']=sample['ramType'].apply(replace_ram_type)
sample['ramSize']=sample['ramSize'].apply(replace_ram_size)
sample['diskType']=sample['diskType'].apply(replace_disk_type)
sample['diskSize']=sample['diskSize'].apply(replace_disk_size)

#Standardize the features
sample=sample.values
sample=std_fit.transform(sample)

#Prediction
if st.button('Predict'):
    price=xgb.predict(sample)
    price=price[0].round(2)    
    st.subheader("Price for these features is estimated around : :blue[{}]".format("₹"+str(price)))
else:
    pass
