import streamlit as st
import pandas as pd
from load_css import local_css
import time
import os

MAX = 30
if "lawdb.csv" in os.listdir():
    f = open("lawdb.csv","a")
else:
    f = open("lawdb.csv",'w')
    f.write('''"name","time","news number","news","articles"\n''')
article_lst = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
'21A', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '32', '32A', '33', '34',
'35', '36', '37', '38', '39', '39A', '40', '41', '42', '43', '43A', '44', '45', '46', '47', '48', '48A', '49', '50',
'51', '51A', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
'70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
'90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107',
'108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124',
'125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '134A', '135', '136', '137', '138', '139', '139A',
'140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156',
'157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173',
'174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190',
'191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207',
'208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
'224A', '225', '226', '227', '228', '229', '230', '231', '233', '233A', '234', '235', '236', '237', '238', '239',
'239A', '239AB', '239B', '240', '241', '243', '243A', '243B', '243C', '243D', '243E', '243F', '243G', '243H', '243I',
'243J', '243K', '243L', '243M', '243N', '243O', '243P', '243Q', '243R', '243S', '243T', '243U', '243V', '243W', '243X',
'243Y', '243Z', '243ZA', '243ZB', '243ZC', '243ZD', '243ZE', '243ZF', '243ZG', '243ZH', '243ZI', '243ZJ', '243ZK',
'243ZL', '243ZM', '243ZN', '243ZO', '243ZP', '243ZQ', '243ZR', '243ZS', '243ZT', '244', '244A', '245', '246', '246A',
'247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '258A', '260', '261', '262', '263',
'264', '265', '266', '267', '268', '268A', '269', '269A', '270', '271', '273', '274', '275', '276', '277', '279', '280',
'281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '290A', '292', '293', '294', '295', '296', '297',
'298', '299', '300', '300A', '320', '301', '302', '303', '304', '305', '307', '308', '309', '310', '311', '312', '312A',
'313', '315', '316', '317', '318', '319', '321', '322', '323', '323A', '323B', '324', '325', '326', '327', '328', '329',
'330', '331', '332', '333', '334', '335', '336', '337', '338', '338A', '339', '340', '341', '342', '343', '344', '345',
'346', '347', '348', '349', '350', '350A', '350B', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360',
'361', '361A', '361B', '363', '363A', '364', '365', '366', '367', '368', '369', '370', '371', '371A', '371B', '371C',
'371D', '371E', '371F', '371G', '371H', '371I', '372', '372A', '373', '374', '375', '376', '377', '378', '378A', '392',
'393', '394', '394A', '395')
st.set_page_config(
    page_title="EQSense",
    page_icon="✌️",
    layout="centered",
)
st.header("Digital Legislation Survey")
local_css("style.css")
backslash_char = "\\"
IN = st.empty()
PBT = st.empty()
PB = st.empty()
IP = st.empty()
FB = st.empty()
NC = st.empty()
st.write("")
st.write("")
IN1 = st.empty()
IN2 = st.empty()
AN = st.empty()
df = pd.read_csv("pyrdf2vec.csv")
app_state = st.experimental_get_query_params()
if "counter" not in app_state.keys():
    counter = 0
    st.experimental_set_query_params(**{'counter': counter})
else:
    counter = int(app_state['counter'][0])

if "lname" not in app_state.keys():
    lname = f"lawyer_{int(time.time())}"
    lname = IP.text_input("Your Name:")
    btnFLG = FB.button("OK")
    if btnFLG:
        if lname.strip() == "":
            st.error("Please enter your name!")
        else:
            st.experimental_set_query_params(**{'lname': lname})
            IN.info(f"Hi {lname}, Please provide your valuable input 😊")
            counter += 1
            st.experimental_set_query_params(**{'lname': lname, 'counter': counter})
            IP.empty()
            FB.empty()

else:
    lname = app_state['lname'][0]

if 0 < counter < MAX+1:
    IN1.info("Which Articles do you think are related with this news? Please select from below!")
    IN2.info("You can select more than one Article.")
    TXT = df['news'].values[counter - 1]
    NEWS = f"<div align='justify: inter-word;><span class='highlight blue'><span class='bold'>NEWS-{counter}: </span>{TXT}</span></div>"
    NC.markdown(NEWS, unsafe_allow_html=True)
    ART_lst = AN.multiselect("Article(s):", article_lst, help="Choose one/multiple articles", key=counter)
    PBT.success(f"{counter-1}/{MAX} Done! ✌️")
    PB.progress((counter-1)/MAX)
    if st.button("Submit"):
        if ART_lst==[]:
            st.error("Please select an article!")
        else:
            TXT = TXT.replace('"',"'")
            f.write(f'''"{lname}",{int(time.time())},"{counter-1}","{TXT}","{ART_lst}"\n''')
            counter += 1
            st.experimental_set_query_params(**{'lname': lname, 'counter': counter})
            TXT = df['news'].values[counter - 1]
            NEWS = f"<div align='justify: inter-word;><span class='highlight blue'><span class='bold'>NEWS-{counter}: </span>{TXT}</span></div>"
            NC.markdown(NEWS, unsafe_allow_html=True)
            ART_lst = AN.multiselect("Article(s):", article_lst, help="Choose one/multiple articles", key=counter)
            PB.progress((counter-1)/MAX)
            PBT.success(f"{counter-1}/{MAX} Done! ✌️")
            if counter==MAX+1:
                NC.empty()
                AN.empty()
                st.balloons()
                st.header("Thank you for your valuable time! 😊")
elif counter>=MAX+1:
    st.balloons()
    st.header("Thank you for your valuable time! 😊")
f.close()