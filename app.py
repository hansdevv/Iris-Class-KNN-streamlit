import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_csv('datasets/Iris.csv')
# unjaya = Image.open(r'assets\img\logo_UNJAYA.png')

st.set_page_config(
	page_title="Iris-class-KNN	",
	page_icon="smile",
	layout="wide",
	initial_sidebar_state="expanded",
	menu_items={
		'Get Help': 'https://www.extremelycoolapp.com/help',
		'Report a bug': "https://www.extremelycoolapp.com/bug",
		'About': "# This is a header. This is an *extremely* cool app!"
	}
)

with st.sidebar:
	# st.image(unjaya,caption="Universitas Jenderal Achmad Yani Yogyakarta", width=270,)
	choose = option_menu(
		"Iris Classification", ["Home", "Classifikasi"],
		icons=['house', 'camera fill'],
		menu_icon="terminal", default_index=0,
		styles={
			"container": {"padding": "1!important", "background-color": "transparent"},
			"icon": {"color": "orange", "font-size": "25px"}, 
			"nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "transparent"},
			"nav-link-selected": {"background-color": "#02ab21"},
		}
	)

if choose == 'Home':
	
	c = st.container()
	col1, col2, col3 = st.columns([1,12,1])
	with col2:
		st.columns(3)[1].title('10 Data Pertama')
		st.write(df.head(10))
	
		st.columns(3)[1].title('10 Data Terakhir')
		st.columns(1,gap="large")[0].write(df.tail(10))

		sub1, sub2 = st.columns([1,3])
		sub1.subheader('Nama Colums')
		sub1.write(df.columns)

		sub2.subheader('Tampilkan data SepalLengthCm dan Species')
		sub2.write(df.loc[:,['SepalLengthCm','Species']])

elif choose == 'Classifikasi':
	#fungsi untuk mencari nilai min columns dataframe
	def min_val(params):
		return float(df[params].min())
	#fungsi untuk mencari nilai max columns dataframe
	def max_val(params):
		return float(df[params].max())
	
	#slider inputan
	st.title('Clasifikasi Iris dengan metode KNN')
	col1, col2=st.columns([3,3])
	with col1:
		SepalLengthCm = col1.slider('SepalLengthCm', min_value=min_val('SepalLengthCm'), max_value=max_val('SepalLengthCm'), )
		SepalWidthCm = col1.slider('SepalWidthCm', min_value=min_val('SepalWidthCm'), max_value=max_val('SepalWidthCm'))
	with col2:
		PetalLengthCm = col2.slider('PetalLengthCm', min_value=min_val('PetalLengthCm'), max_value=max_val('PetalLengthCm'), )
		PetalWidthCm = col2.slider('PetalWidthCm', min_value=min_val('PetalWidthCm'), max_value=max_val('PetalWidthCm'))

	#proses clasifikasi saat tombol diclick
	if st.button('submit'):
		X = df.drop(['Id', 'Species'], axis=1)
		y = df['Species']
		x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 42)
		knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
		knn.fit(x_train, y_train)
		iris_class = knn.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])

		st.subheader('Kasifikasi termasuk bunga')
		st.success(iris_class[0])
	else:
		st.write('')