import streamlit as st
import pandas as pd
from load_css import local_css

article_lst = ['1', '2', '2A', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
               '19', '20', '21', '21A', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '31A', '31B', '31C',
               '31D', '32', '32A', '33', '34', '35', '36', '37', '38', '39', '39A', '40', '41', '42', '43', '43A',
               '43-B', '44', '45', '46', '47', '48', '48A', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
               '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
               '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92',
               '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108',
               '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123',
               '124', '124A', '124B', '124C', '125', '126', '127', '128', '129', '130', '131', '131A', '132', '133',
               '134', '134A', '135', '136', '137', '138', '139', '139A', '140', '141', '142', '143', '144', '144A',
               '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
               '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174',
               '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189',
               '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204',
               '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219',
               '220', '221', '222', '223', '224', '224A', '225', '226', '226A', '227', '228', '228A', '229', '230',
               '231', '233', '233A', '234', '235', '236', '237', '239', '239A', '239AA', '239AB', '239B', '240', '241',
               '242', '243', '243A', '243B', '243C', '243D', '243E', '243F', '243G', '243H', '243-I', '243J', '243K',
               '243L', '243M', '243N', '243-O', '243P', '243Q', '243R', '243S', '243T', '243U', '243V', '243W', '243X',
               '243Y', '243Z', '243ZA', '243ZB', '243ZC', '243ZD', '243ZE', '243ZF', '243ZG', '244', '244A', '245',
               '246', '246A', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '257A',
               '258', '258A', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '269A',
               '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '279A', '280', '281', '282', '283',
               '284', '285', '286', '287', '288', '289', '290', '290A', '291', '292', '293', '294', '295', '296', '297',
               '298', '299', '300', '300A', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311',
               '312', '312A', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '323A',
               '323B', '324', '325', '326', '327', '328', '329', '329A', '330', '331', '332', '333', '334', '335',
               '336', '337', '338', '338A', '338B', '339', '340', '341', '342', '343', '344', '345', '346', '347',
               '348', '349', '350', '350A', '350B', '351', '352', '353', '354', '355', '356', '357', '358', '359',
               '359A', '360', '361', '361A', '361B', '362', '363', '363A', '364', '365', '366', '367', '369', '371',
               '371A', '371B', '371C', '371D', '371E', '371F', '371G', '371H', '371-I', '371J', '372', '372A', '373',
               '374', '375', '376', '377', '378', '378A', '379—', '392', '393', '394', '394A', '395']
st.set_page_config(
    page_title="EQSense",
    page_icon="✌️",
    layout="centered",
)
local_css("style.css")
backslash_char = "\\"
METHOD = st.sidebar.radio("", ("RDF2Vec", "Word2Vec", "FastText/BERT/GPT"))
if METHOD == "RDF2Vec":
    df = pd.read_csv("pyrdf2vec.csv")
    for i in range(df.shape[0]):
        # st.write(df['news'].values[i])
        # st.write(" ".join(eval(df['articles'].values[i])))
        # st.markdown("---")

        NEWS = f"<div align='justify: inter-word;><span class='highlight blue'><span class='bold'>NEWS-{i + 1}: </span>{df['news'].values[i]}</span></div>"
        st.markdown(NEWS, unsafe_allow_html=True)
        buff = eval(df['articles'].values[i])
        counter = 0
        st.write("")
        st.markdown("<b>ARTICLES</b>", unsafe_allow_html=True)
        for j in buff:
            if counter == 0:
                locals()[f'{i}_CM_{j}'] = st.checkbox(j, value=True, key=f"{i}_{j}")
            else:
                locals()[f'{i}_CM_{j}'] = st.checkbox(j, value=False, key=f"{i}_{j}")
            counter += 1
        locals()[f"{i}_CM_OTHERS"] = st.multiselect("Other:", article_lst, help = "Choose one/multiple articles", key=i)
        # ARTICLES = f"<div align='right'><span class='highlight red'><span class='bold'>ARTICLES: </span>{buff}</span></div>"
        st.markdown("---")
elif METHOD=="Word2Vec":
    df = pd.read_csv("word2vec.csv")
    for i in range(df.shape[0]):
        # st.write(df['news'].values[i])
        # st.write(" ".join(eval(df['articles'].values[i])))
        # st.markdown("---")

        NEWS = f"<div align='justify: inter-word;><span class='highlight blue'><span class='bold'>NEWS-{i + 1}: </span>{df['news'].values[i]}</span></div>"
        st.markdown(NEWS, unsafe_allow_html=True)
        buff = eval(df['articles'].values[i])
        counter = 0
        st.write("")
        st.markdown("<b>ARTICLES</b>", unsafe_allow_html=True)
        for j in buff:
            if counter == 0:
                locals()[f'{i}_CM_{j}'] = st.checkbox(j, value=True, key=f"{i}_{j}")
            else:
                locals()[f'{i}_CM_{j}'] = st.checkbox(j, value=False, key=f"{i}_{j}")
            counter += 1
        locals()[f"{i}_CM_OTHERS"] = st.multiselect("Other:", article_lst, help="Choose one/multiple articles", key=i)
        # ARTICLES = f"<div align='right'><span class='highlight red'><span class='bold'>ARTICLES: </span>{buff}</span></div>"
        st.markdown("---")
else:
    st.title("Coming Soon")
