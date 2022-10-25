import pickle

import numpy as np
import pandas as pd


class ClusterChain:
    def __init__(self, chapters, blocks, categories, codes, chain):
        self.chapter = chapters
        self.blocks = blocks
        self.categories = categories
        self.codes = codes
        # matrix list
        self.chain = chain
        self.levels = len(chain)

label_dict = pd.read_csv('../data/mimic3/full/labels_dictionary_full_raw.csv')

# print(label_dict)
print('load labels from file')

def generate_category(code):
    return code.split('.')[0]


def generate_block(code):
    code = code.split('.')[0]

    if len(code) != 2:
        if code.isdigit():
            value = int(code)
            if 139 >= value >= 1:
                if 9 >= value >= 1:
                    return 'B011'
                elif 18 >= value >= 10:
                    return 'B012'
                elif 27 >= value >= 19:
                    return 'B013'
                elif 41 >= value >= 30:
                    return 'B014'
                elif 42 == value:
                    return 'B015'
                elif 49 >= value >= 45:
                    return 'B016'
                elif 59 >= value >= 50:
                    return 'B017'
                elif 66 >= value >= 60:
                    return 'B018'
                elif 79 >= value >= 70:
                    return 'B019'
                elif 88 >= value >= 80:
                    return 'B0110'
                elif 99 >= value >= 90:
                    return 'B0111'
                elif 104 >= value >= 100:
                    return 'B0112'
                elif 118 >= value >= 110:
                    return 'B0113'
                elif 129 >= value >= 120:
                    return 'B0114'
                elif 136 >= value >= 130:
                    return 'B0115'
                elif 139 >= value >= 137:
                    return 'B0116'
            elif 239 >= value >= 140:
                if 149 >= value >= 140:
                    return 'B021'
                elif 159 >= value >= 150:
                    return 'B022'
                elif 165 >= value >= 160:
                    return 'B023'
                elif 176 >= value >= 170:
                    return 'B024'
                elif 189 >= value >= 179:
                    return 'B025'
                elif 199 >= value >= 190:
                    return 'B026'
                elif 209 >= value >= 200:
                    return 'B027'
                elif 229 >= value >= 210:
                    return 'B028'
                elif 234 >= value >= 230:
                    return 'B029'
                elif 238 >= value >= 235:
                    return 'B0210'
                elif 239 == value:
                    return 'B0211'
            elif 279 >= value >= 240:
                if 246 >= value >= 240:
                    return 'B031'
                elif 259 >= value >= 249:
                    return 'B032'
                elif 269 >= value >= 260:
                    return 'B033'
                elif 279 >= value >= 270:
                    return 'B034'
            elif 289 >= value >= 280:
                if 280 == value:
                    return 'B041'
                elif 281 == value:
                    return 'B042'
                elif 282 == value:
                    return 'B043'
                elif 283 == value:
                    return 'B044'
                elif 284 == value:
                    return 'B045'
                elif 285 == value:
                    return 'B046'
                elif 286 == value:
                    return 'B047'
                elif 287 == value:
                    return 'B048'
                elif 288 == value:
                    return 'B049'
                elif 289 == value:
                    return 'B0410'
            elif 319 >= value >= 290:
                if 294 >= value >= 290:
                    return 'B051'
                elif 299 >= value >= 295:
                    return 'B052'
                elif 316 >= value >= 300:
                    return 'B053'
                elif 319 >= value >= 317:
                    return 'B054'
            elif 389 >= value >= 320:
                if 327 >= value >= 320:
                    return 'B061'
                elif 337 >= value >= 330:
                    return 'B062'
                elif 338 == value:
                    return 'B063'
                elif 339 == value:
                    return 'B064'
                elif 349 >= value >= 340:
                    return 'B065'
                elif 359 >= value >= 350:
                    return 'B066'
                elif 379 >= value >= 360:
                    return 'B067'
                elif 389 >= value >= 380:
                    return 'B068'
            elif 459 >= value >= 390:
                if 392 >= value >= 390:
                    return 'B071'
                elif 398 >= value >= 393:
                    return 'B072'
                elif 405 >= value >= 401:
                    return 'B073'
                elif 414 >= value >= 410:
                    return 'B074'
                elif 417 >= value >= 415:
                    return 'B075'
                elif 429 >= value >= 420:
                    return 'B076'
                elif 438 >= value >= 430:
                    return 'B077'
                elif 449 >= value >= 440:
                    return 'B078'
                elif 459 >= value >= 451:
                    return 'B079'
            elif 519 >= value >= 460:
                if 466 >= value >= 460:
                    return 'B081'
                elif 478 >= value >= 470:
                    return 'B082'
                elif 488 >= value >= 480:
                    return 'B083'
                elif 496 >= value >= 490:
                    return 'B084'
                elif 508 >= value >= 500:
                    return 'B085'
                elif 519 >= value >= 510:
                    return 'B086'
            elif 579 >= value >= 520:
                if 529 >= value >= 520:
                    return 'B091'
                elif 539 >= value >= 530:
                    return 'B092'
                elif 543 >= value >= 540:
                    return 'B093'
                elif 553 >= value >= 550:
                    return 'B094'
                elif 558 >= value >= 555:
                    return 'B095'
                elif 569 >= value >= 560:
                    return 'B096'
                elif 579 >= value >= 570:
                    return 'B097'
            elif 629 >= value >= 580:
                if 589 >= value >= 580:
                    return 'B101'
                elif 599 >= value >= 590:
                    return 'B102'
                elif 608 >= value >= 600:
                    return 'B103'
                elif 612 >= value >= 610:
                    return 'B104'
                elif 616 >= value >= 614:
                    return 'B105'
                elif 629 >= value >= 617:
                    return 'B106'
            elif 679 >= value >= 630:
                if 639 >= value >= 630:
                    return 'B111'
                elif 649 >= value >= 640:
                    return 'B112'
                elif 659 >= value >= 650:
                    return 'B113'
                elif 669 >= value >= 660:
                    return 'B114'
                elif 677 >= value >= 670:
                    return 'B115'
                elif 679 >= value >= 678:
                    return 'B116'
            elif 709 >= value >= 680:
                if 686 >= value >= 680:
                    return 'B121'
                elif 698 >= value >= 690:
                    return 'B122'
                elif 709 >= value >= 700:
                    return 'B123'
            elif 739 >= value >= 710:
                if 719 >= value >= 710:
                    return 'B131'
                elif 724 >= value >= 720:
                    return 'B132'
                elif 729 >= value >= 725:
                    return 'B133'
                elif 739 >= value >= 730:
                    return 'B134'
            elif 759 >= value >= 740:
                if 740 == value:
                    return 'B141'
                elif 741 == value:
                    return 'B142'
                elif 742 == value:
                    return 'B143'
                elif 743 == value:
                    return 'B144'
                elif 744 == value:
                    return 'B145'
                elif 745 == value:
                    return 'B146'
                elif 746 == value:
                    return 'B147'
                elif 747 == value:
                    return 'B148'
                elif 748 == value:
                    return 'B149'
                elif 749 == value:
                    return 'B1410'
                elif 750 == value:
                    return 'B1411'
                elif 751 == value:
                    return 'B1412'
                elif 752 == value:
                    return 'B1413'
                elif 753 == value:
                    return 'B1414'
                elif 754 == value:
                    return 'B1415'
                elif 755 == value:
                    return 'B1416'
                elif 756 == value:
                    return 'B1417'
                elif 757 == value:
                    return 'B1418'
                elif 758 == value:
                    return 'B1419'
                elif 759 == value:
                    return 'B1420'
            elif 779 >= value >= 760:
                if 763 >= value >= 760:
                    return 'B151'
                elif 779 >= value >= 764:
                    return 'B152'
            elif 799 >= value >= 780:
                if 789 >= value >= 780:
                    return 'B161'
                elif 796 >= value >= 790:
                    return 'B162'
                elif 799 >= value >= 797:
                    return 'B163'
            elif 999 >= value >= 800:
                if 804 >= value >= 800:
                    return 'B171'
                elif 809 >= value >= 805:
                    return 'B172'
                elif 819 >= value >= 810:
                    return 'B173'
                elif 829 >= value >= 820:
                    return 'B174'
                elif 839 >= value >= 830:
                    return 'B175'
                elif 848 >= value >= 840:
                    return 'B176'
                elif 854 >= value >= 850:
                    return 'B177'
                elif 869 >= value >= 860:
                    return 'B178'
                elif 879 >= value >= 870:
                    return 'B179'
                elif 887 >= value >= 880:
                    return 'B180'
                elif 897 >= value >= 890:
                    return 'B181'
                elif 904 >= value >= 900:
                    return 'B182'
                elif 909 >= value >= 905:
                    return 'B183'
                elif 919 >= value >= 910:
                    return 'B184'
                elif 924 >= value >= 920:
                    return 'B185'
                elif 929 >= value >= 925:
                    return 'B186'
                elif 939 >= value >= 930:
                    return 'B187'
                elif 949 >= value >= 940:
                    return 'B188'
                elif 957 >= value >= 950:
                    return 'B189'
                elif 959 >= value >= 958:
                    return 'B190'
                elif 979 >= value >= 960:
                    return 'B191'
                elif 989 >= value >= 980:
                    return 'B192'
                elif 995 >= value >= 990:
                    return 'B193'
                elif 999 >= value >= 996:
                    return 'B194'
            else:
                print("Diagnosis: {}".format(code))
        else:
            if code.startswith("E"):
                value = int(code[1:])
                if 0 == value:
                    return 'BE1'
                elif 30 >= value >= 1:
                    return 'BE2'
                elif 807 >= value >= 800:
                    return 'BE3'
                elif 819 >= value >= 810:
                    return 'BE4'
                elif 825 >= value >= 820:
                    return 'BE5'
                elif 829 >= value >= 826:
                    return 'BE6'
                elif 838 >= value >= 830:
                    return 'BE7'
                elif 845 >= value >= 840:
                    return 'BE8'
                elif 849 >= value >= 846:
                    return 'BE9'
                elif 858 >= value >= 850:
                    return 'BE10'
                elif 869 >= value >= 860:
                    return 'BE11'
                elif 876 >= value >= 870:
                    return 'BE12'
                elif 879 >= value >= 878:
                    return 'BE13'
                elif 888 >= value >= 880:
                    return 'BE14'
                elif 899 >= value >= 890:
                    return 'BE15'
                elif 909 >= value >= 900:
                    return 'BE16'
                elif 915 >= value >= 910:
                    return 'BE17'
                elif 928 >= value >= 916:
                    return 'BE18'
                elif 929 == value:
                    return 'BE19'
                elif 949 >= value >= 930:
                    return 'BE20'
                elif 959 >= value >= 950:
                    return 'BE21'
                elif 969 >= value >= 960:
                    return 'BE22'
                elif 979 >= value >= 970:
                    return 'BE23'
                elif 989 >= value >= 980:
                    return 'BE24'
                elif 999 >= value >= 990:
                    return 'BE25'
            elif code.startswith("V"):
                value = int(code[1:])
                if 9 >= value >= 1:
                    return 'BV1'
                elif 19 >= value >= 10:
                    return 'BV2'
                elif 29 >= value >= 20:
                    return 'BV3'
                elif 39 >= value >= 30:
                    return 'BV4'
                elif 49 >= value >= 40:
                    return 'BV5'
                elif 59 >= value >= 50:
                    return 'BV6'
                elif 69 >= value >= 60:
                    return 'BV7'
                elif 82 >= value >= 70:
                    return 'BV8'
                elif 84 >= value >= 83:
                    return 'BV9'
                elif 85 == value:
                    return 'BV10'
                elif 86 == value:
                    return 'BV11'
                elif 87 == value:
                    return 'BV12'
                elif 88 == value:
                    return 'BV13'
                elif 89 == value:
                    return 'BV14'
                elif 90 == value:
                    return 'BV15'
                elif 91 == value:
                    return 'BV16'
            else:
                print("Diagnosis: {}".format(code))
                return "B0"
    else:  # Procedure Codes http://www.icd9data.com/2012/Volume3/default.htm
        if code.isdigit():
            value = int(code)
            if value == 0:
                return "BP1" + str(value)
            elif 5 >= value >= 1:
                return "BP2" + str(value)
            elif 7 >= value >= 6:
                return "BP3" + str(value)
            elif 16 >= value >= 8:
                return "BP4" + str(value)
            elif 17 >= value >= 17:
                return "BP5" + str(value)
            elif 20 >= value >= 18:
                return "BP6" + str(value)
            elif 29 >= value >= 21:
                return "BP7" + str(value)
            elif 34 >= value >= 30:
                return "BP8" + str(value)
            elif 39 >= value >= 35:
                return "BP9" + str(value)
            elif 41 >= value >= 40:
                return "BP10" + str(value)
            elif 54 >= value >= 42:
                return "BP11" + str(value)
            elif 59 >= value >= 55:
                return "BP12" + str(value)
            elif 64 >= value >= 60:
                return "BP13" + str(value)
            elif 71 >= value >= 65:
                return "BP14" + str(value)
            elif 75 >= value >= 72:
                return "BP15" + str(value)
            elif 84 >= value >= 76:
                return "BP16" + str(value)
            elif 86 >= value >= 85:
                return "BP17" + str(value)
            elif 99 >= value >= 87:
                return "BP18" + str(value)
            else:
                print("Procedure: {}".format(code))
                return 'BP0'
        else:
            print("Procedure: {}".format(code))
            return 'BP0'


def generate_chapter(code):
    # if '.' not in code:
    #     print(code)
    code = code.split('.')[0]
    if len(code) != 2:
        if code.isdigit():
            value = int(code)
            if 139 >= value >= 1:
                return "D1"
            elif 239 >= value >= 140:
                return "D2"
            elif 279 >= value >= 240:
                return "D3"
            elif 289 >= value >= 280:
                return "D4"
            elif 319 >= value >= 290:
                return "D5"
            elif 389 >= value >= 320:
                return "D6"
            elif 459 >= value >= 390:
                return "D7"
            elif 519 >= value >= 460:
                return "D8"
            elif 579 >= value >= 520:
                return "D9"
            elif 629 >= value >= 580:
                return "D10"
            elif 679 >= value >= 630:
                return "D11"
            elif 709 >= value >= 680:
                return "D12"
            elif 739 >= value >= 710:
                return "D13"
            elif 759 >= value >= 740:
                return "D14"
            elif 779 >= value >= 760:
                return "D15"
            elif 799 >= value >= 780:
                return "D16"
            elif 999 >= value >= 800:
                return "D17"
            else:
                print("Diagnosis: {}".format(code))
        else:
            if code.startswith("E") or code.startswith("V"):
                return "D18"
            else:
                print("Diagnosis: {}".format(code))
                return "D0"
    else:  # Procedure Codes http://www.icd9data.com/2012/Volume3/default.htm
        if code.isdigit():
            value = int(code)
            if value == 0:
                return "P1"
            elif 5 >= value >= 1:
                return "P2"
            elif 7 >= value >= 6:
                return "P3"
            elif 16 >= value >= 8:
                return "P4"
            elif 17 >= value >= 17:
                return "P5"
            elif 20 >= value >= 18:
                return "P6"
            elif 29 >= value >= 21:
                return "P7"
            elif 34 >= value >= 30:
                return "P8"
            elif 39 >= value >= 35:
                return "P9"
            elif 41 >= value >= 40:
                return "P10"
            elif 54 >= value >= 42:
                return "P11"
            elif 59 >= value >= 55:
                return "P12"
            elif 64 >= value >= 60:
                return "P13"
            elif 71 >= value >= 65:
                return "P14"
            elif 75 >= value >= 72:
                return "P15"
            elif 84 >= value >= 76:
                return "P16"
            elif 86 >= value >= 85:
                return "P17"
            elif 99 >= value >= 87:
                return "P18"
            else:
                print("Procedure: {}".format(code))
                return 'P0'
        else:
            print("Procedure: {}".format(code))
            return 'P0'


label_dict['category'] = label_dict['icd9_code'].apply(generate_category)
label_dict['block'] = label_dict['icd9_code'].apply(generate_block)
label_dict['chapter'] = label_dict['icd9_code'].apply(generate_chapter)

# label_dict.to_pickle('../data/hi_label_tree/hlt_df.pkl')
unique_code = label_dict['icd9_code'].unique()
unique_category = label_dict['category'].unique()
unique_block = label_dict['block'].unique()
unique_chapter = label_dict['chapter'].unique()
unique_chapter.sort()
unique_block.sort()
unique_category.sort()
unique_code.sort()
print('Chapter {}'.format(len(unique_chapter)))
print('Block {}'.format(len(unique_block)))
print('Category {}'.format(len(unique_category)))
print('Code {}'.format(len(unique_code)))
unique_chapter.sort()

# create indexing matrices
C_chapters = np.ones((len(unique_chapter), 1))  # k1*k0 k0=1
C_blocks = np.zeros((len(unique_block), len(unique_chapter)))
C_categories = np.zeros((len(unique_category), len(unique_block)))
C_codes = np.zeros((len(unique_code), len(unique_category)))  # k4*k3 k4=L
# create chain of hierarchical label tree
cluster_chain = []
cluster_chain.append(C_chapters)
# create matrix of block
temp_df = label_dict[["chapter", "block"]]
temp_df = temp_df.drop_duplicates()
for idx, row in temp_df.iterrows():
    chapter_idx = np.where(unique_chapter == row['chapter'])[0][0]
    block_idx = np.where(unique_block == row['block'])[0][0]
    C_blocks[block_idx][chapter_idx] = 1
cluster_chain.append(C_blocks)

# create matrix of category
temp_df = label_dict[["block", "category"]]
temp_df = temp_df.drop_duplicates()
for idx, row in temp_df.iterrows():
    block_idx = np.where(unique_block == row['block'])[0][0]
    category_idx = np.where(unique_category == row['category'])[0][0]
    C_categories[category_idx][block_idx] = 1
cluster_chain.append(C_categories)

# create matrix of code
temp_df = label_dict[["category", "icd9_code"]]
temp_df = temp_df.drop_duplicates()
for idx, row in temp_df.iterrows():
    category_idx = np.where(unique_category == row['category'])[0][0]
    code_idx = np.where(unique_code == row['icd9_code'])[0][0]
    C_codes[code_idx][category_idx] = 1
cluster_chain.append(C_codes)

print('Hierarchical code tree created.')

with open('../data/mimic3/hct.pkl', 'wb') as f:
    pickle.dump(cluster_chain, f, protocol=pickle.HIGHEST_PROTOCOL)

