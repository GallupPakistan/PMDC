import pandas as pd
from pymongo import MongoClient
import streamlit as st


@st.cache_resource
def get_mongo_db():
    client = MongoClient(st.secrets["MONGO_URI"])
    return client[st.secrets["DATABASE"]]


def get_mongo_collection(collection_name):
    db = get_mongo_db()
    return db[collection_name]