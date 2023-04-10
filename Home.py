import streamlit as st
from PIL import Image


st.image(Image.open('cat.jpg'))

st.markdown("<h2 style='text-align: center;'>Hi!</h2>", unsafe_allow_html=True)

cont=st.container()
cont.markdown("<h2 style='text-align: center; color:#1189cf'>I'm Shubham Yadav</h2>", unsafe_allow_html=True)
cont.subheader("I'm a Data Science Intern at :red[Innomatics Research Labs] and an aspiring Data scientist with a passion for exploring and analyzing datasets to derive valuable insights from them.",anchor='#')
for i in range(5):st.write("")

c1,c2,c3,c4=st.columns(4)
c2.subheader("[LinkedIn](https://www.linkedin.com/in/shubham-yadav-24153a152//streamlit_app.py)") 
c3.subheader("[Github](https://github.com/ZealousGuy)")

cont.subheader('Get to know me :point_down:')