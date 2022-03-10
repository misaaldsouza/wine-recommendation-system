# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle5 as pickle
import streamlit as st
import cssutils
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from PIL import Image

# Path of Application Folder
path = "C:/Users/misaa/OneDrive/Desktop/WineRecommendationSystem/"
df = pd.read_csv(path + "wine.csv")

col = ['province','variety','points']
wine1 = df[col]
wine1 = wine1.dropna(axis=0)
wine1 = wine1.drop_duplicates(['province','variety'])
wine1 = wine1[wine1['points'] >85]
wine_pivot = wine1.pivot(index= 'variety',columns='province',values='points').fillna(0)
wine_pivot_matrix = csr_matrix(wine_pivot)

wine_pivot_find = wine_pivot.copy()

wine_pivot_find.reset_index(level=0, inplace=True)

# Model
knn = NearestNeighbors(n_neighbors=10,algorithm= 'brute', metric= 'cosine')
model_knn = knn.fit(wine_pivot_matrix)

# Main Function
results = []
def wine_recommendation(var):
 query_index = wine_pivot_find.index[wine_pivot_find.variety == var]
 arr = query_index.values
 for i in range(1):
     query_index = arr[i]
     distance, indice = model_knn.kneighbors(wine_pivot.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
     for i in range(0, len(distance.flatten())):
        if i!=0:
            x = i,wine_pivot.index[indice.flatten()[i]]
            results.append(x[1])
     return results

def predict_wrs(var):
    one = 'https://www.google.com/search?q='
    r = wine_recommendation(var)
    st.subheader("Recommednations for "+var+":\n")
    for i in range(0,5):
        x = r[i]
        y = x # With spaces
        if ' ' in y == True :
            y = y.replace(' ','+')
        link = one + y.replace(' ','+')
        st.write('* [{}]({})'.format(y,link))

# Pickle : write and read
pickle_out = open(path+"predict_wrs.pkl", "wb")
pickle.dump(predict_wrs, pickle_out)
pickle_out.close()

pickle_in = open(path+'predict_wrs.pkl', 'rb')
classifier = pickle.load(pickle_in)

# Input
st.title('Wine Recommendation System')
st.subheader('Variety Name:')
wine_name = st.text_input('')
submit = st.button('Predict')
if submit:
  predict_wrs(wine_name)

# Information
import pandas as pd
import streamlit as st
st.subheader('Information')
st.write('* Ranking Of Main Grape Varieties Planted Worldwide')
df = pd.DataFrame({
    "Grape": [
                'Kyohō (dessert grape)','Cabernet-Sauvignon','Sultanine (wine, dessert and raisin grape)',
                'Merlot (wine grape)', 'Tempranillo (wine grape)', 'Airen (wine and distillation grape)',
                'Chardonnay (wine grape)', 'Syrah (wine grape)', 'Grenache noir (or Garnacha tinta)',
                'Red Globe (dessert grape)'],
    "Hectares": [365000,340000,30000,266000,231000,218000,211000,190000,163000,160000]})


#
# set index to empty strings
df.index = [""] * len(df)
st.table(df)


# Bordeaux
st.write("* Bordeaux-style Red Blend")
col1, mid, col2 = st.beta_columns([3,1,14])
with col1:
    st.image('images/bordeaux.jpg', width=125)
with col2:
    st.write("Bordeaux type of grapes are used for the most expensive wines.")
# Portuguese Red
st.write("* Portuguese Red")
col1, mid, col2 = st.beta_columns([3,1,14])
with col1:
    st.image('images/portuguese.jpg', width=125)
with col2:
    st.write("Portuguese grapes are found to be used in the wines that have the highest ratings from the wine tasters.")












# image
st.subheader('Visualization')
# viz 01
st.write('* Most & Least Expensive Wine by Country')
img = Image.open("images/most_exp.png")
st.image(img)
# viz 02
st.write('* Most Expensive & Most Rated Wine Prepared by Country')
img = Image.open("images/most_prep.png")
st.image(img)
# viz 03
st.write('* Wine Taster Count by Country')
img = Image.open("images/taster_country.png")
st.image(img)
# viz 04
st.write('* Common Words Used to Describe Wine')
img = Image.open("images/common_words.png")
st.image(img)



# Author
st.subheader('Top Wine Tasters In the World & their Reviews')
# kerin
import streamlit as st
st.write("**Kerin O’Keefe**")
col1, mid, col2 = st.beta_columns([3,1,14])
with col1:
    st.image('images/kerin.jpg', width=125)
with col2:
    st.write("Kerin O’Keefe is a wine critic, writer and public speaker, specialising in Italian wine. She has tasted 10,776 wines from all over the world. She has been acknowledged to be one of the great wine commentators on Italy")
    st.write("""Review on White Blend : *"Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.*""")
# roger
st.write("**Roger Voss**")
# 1 1 20
col1, mid, col2 = st.beta_columns([3,1,14])
with col1:
    st.image('images/roger.jpg', width=125)
with col2:
    st.write("Roger Voss is a veteran wine and food author, and a journalist and has been writing about wine and food for the past 25 years. He has tasted 25,514 wines till date.")
    st.write("""Review on Chardonnay : *"This soft, rounded wine is ripe with generous pear and melon flavors. It's easy and ready to drink young for its smooth, attractively ripe character."*""")
# michael
st.write("**Michael Schachner**")
# 1 1 20
col1, mid, col2 = st.beta_columns([3,1,14])
with col1:
    st.image('images/michae.jpg', width=125)
with col2:
    st.write("Michael Schachner is a New York-based journalist specializing in wine, food and travel. His areas of wine expertise include Spain and South America. He has tasted 15,134 wines till date.")
    st.write("""Review on Pinot Noir : *"Lightly herbal strawberry and raspberry aromas are authentic and fresh. On the palate, this is light and juicy, with snappy, lean flavors of red fruit and dry spice. The finish is dry and oaky."*""")
# virginie
# michael
st.write("**Virginie Boone**")
col1, mid, col2 = st.beta_columns([3,1,14])
with col1:
    st.image('images/virginie.jpeg', width=125)
with col2:
    st.write("Virginie Boone reviews and writes about the wines of Napa and Sonoma for Wine Enthusiast Media. She has tasted 9,534 wines till date.")
    st.write("""Review on Cabernet Sauvignon: *"Soft, supple plum envelopes an oaky structure in this Cabernet, supported by 15% Merlot. Coffee and chocolate complete the picture, finishing strong at the end, resulting in a value-priced wine of attractive flavor and immediate accessibility."*""")

# Background Image
import base64
main_bg = "images/bg.png"
main_bg_ext = "png"
side_bg = "images/bg.png"
side_bg_ext = "png"
st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/gif/video{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
       .sidebar .sidebar-content {{
            background: url(data:image/gif/video{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True
)

# SIDEBAR
st.sidebar.title('Observations')
st.sidebar.write('* France produces nearly every strain of international grape and it also harbors the most [*expensive wine*](https://www.wine-searcher.com/find/dom+leroy+grand+cru+musigny+chambolle+cote+de+nuit+burgundy+france) in the world.')
# sidebar image
st.sidebar.image("images/france.jpg", use_column_width=True)

st.sidebar.write('* US is the highest producer of wine in the world with a count of 54,504 followed by France with 22,093')
st.sidebar.write("""* [*Cabernet Sauvignon*](https://www.google.com/search?q=Cabernet+Sauvignon&rlz=1C1CHBF_enIN912IN912&oq=Cabernet+Sauvignon&aqs=chrome..69i57.239j0j9&sourceid=chrome&ie=UTF-8),
                       [*Pinot Noir*](https://www.google.com/search?q=Pinot+Noir&rlz=1C1CHBF_enIN912IN912&sxsrf=AOaemvLL4jIk2RgsCbwQnAMO2TiKqIqKnA%3A1633078861307&ei=Tc5WYeeUEtC-3LUP3JKamA0&ved=0ahUKEwin76vr7KjzAhVQH7cAHVyJBtMQ4dUDCA8&uact=5&oq=Pinot+Noir&gs_lcp=Cgdnd3Mtd2l6EAMyBAgjECcyBAgAEEMyBAgAEEMyCggAEIAEEIcCEBQyBAgAEEMyBAgAEEMyCAgAEIAEEMkDMgQIABBDMgUIABCABDIFCAAQgAQ6BwgAEEcQsAM6BwgAELADEEM6DQguEMgDELADEEMQkwI6CgguEMgDELADEEM6BAguEEM6CwguEIAEEMcBEKMCOgsIABCABBCxAxCDAToHCC4QsQMQQzoKCC4QsQMQgwEQQzoICC4QgAQQsQNKBQg4EgExSgQIQRgAULbBA1i_zANgy80DaANwAngAgAG1AYgBvgqSAQMwLjmYAQCgAQHIAQ_AAQE&sclient=gws-wiz),
                       [*Airen*](https://www.google.com/search?q=Airen&rlz=1C1CHBF_enIN912IN912&sxsrf=AOaemvKcPAcPUbSrCSRVtu2Ut3mgUK3tNQ%3A1633078921747&ei=ic5WYf-BLcjhz7sPh7iqqAI&ved=0ahUKEwi_6pSI7ajzAhXI8HMBHQecCiUQ4dUDCA8&uact=5&oq=Airen&gs_lcp=Cgdnd3Mtd2l6EAMyBQgAEJECMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDILCC4QgAQQxwEQrwEyBQgAEIAEMgUILhCABDIHCAAQsQMQCjIFCAAQgAQ6BwgjELADECc6BwgAEEcQsAM6CgguEMgDELADEEM6BAgjECc6EQguEIAEELEDEIMBEMcBENEDOgsIABCABBCxAxCDAToICAAQgAQQsQM6EAguELEDEMcBENEDEEMQkwI6BAgAEEM6DgguEIAEELEDEMcBEKMCOggILhCABBCxAzoHCAAQsQMQQzoLCC4QgAQQxwEQ0QM6CAgAEIAEEMkDOgUIABCSA0oFCDgSATFKBAhBGABQh-UCWMvvAmCm8QJoBHACeACAAa4BiAGQBpIBAzAuNZgBAKABAcgBD8ABAQ&sclient=gws-wiz),
                       [*Merlot*](https://www.google.com/search?rlz=1C1CHBF_enIN912IN912&sxsrf=AOaemvLQBcghTBYOqZ_vgiHMnFU9ow_GDQ:1633078970599&q=Merlot&stick=H4sIAAAAAAAAAONQFuLUz9U3MDGsqCpU4gAzi4sMTjEiRE8xgoUtk9MLoMLGFlUFlXDhgjKosKFZSVUelG1UaZlUDlWCZKBhslGV2S9GhaDUnMSS1BSFknwFx8yiwyvzFBLzUhQCMvPySxTy8jOLdrEg7IeywVoXsbL5phbl5JdMYGO8xSbJcGMml2nEg4nXYwXvBb_bGa0w7fvmD9IzrWcBAAVZPvLXAAAA&sa=X)
                       are the most widely used grape varieties in the world.""")
st.sidebar.write('* The countries that produce the cheapest wines include Ukraine, India, Armenia, Blugeria and Bosnia and Herzegovina.')
st.sidebar.write('* [*Roger Voss*](https://www.winemag.com/contributor/roger-voss/) is a food author and has extensive experience of wines from all over the world . He has tasted 25,514 wines till date.')
st.sidebar.write('* [*Verite*](https://www.google.com/maps/place/Verit%C3%A9+Winery/@38.6141926,-122.7714187,17z/data=!3m1!4b1!4m5!3m4!1s0x808414546b9131d5:0xe3ce85105ff6e1e5!8m2!3d38.6139648!4d-122.7689927?hl=en) is the most rated wine prepared in the winery.')
st.sidebar.image('images/verite.jpg')
