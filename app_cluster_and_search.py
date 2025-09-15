import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import ast, re, os

st.set_page_config(
    page_title="Restaurant Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


DF_PATH = "restaurants_with_clusters.pkl" 
TFIDF_REV_PATH = "X_reviews.pkl"
TFIDF_CUIS_PATH = "X_Cuisine.pkl"



def parse_cuisine_field(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                return [str(i).strip() for i in lst if str(i).strip()]
        except Exception:
            pass
    if re.search(r'[;,/|]', s):
        parts = re.split(r'[;,/|]+', s)
        return [p.strip() for p in parts if p.strip()]
    if ' & ' in s or ' and ' in s:
        parts = re.split(r'\s+&\s+|\s+and\s+', s)
        return [p.strip() for p in parts if p.strip()]
    return [s]

# ---------------------------
# load dataframe
# ---------------------------
@st.cache_data(show_spinner=True)
def load_df(path=DF_PATH):
    if not os.path.exists(path):
        st.error(f"Data file not found at {path}. Place restaurants_with_clusters.pkl in the app folder.")
        return pd.DataFrame()
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    

df = load_df(DF_PATH)
if df.empty:
    st.stop()

# pre-parse cuisine lists (cached)
@st.cache_data
def get_cuisine_lists(series):
    return series.apply(parse_cuisine_field)

df['cuisine_list'] = get_cuisine_lists(df['cuisine'])


st.sidebar.header("Mode & Filters")
mode = st.sidebar.radio("Mode", ("Cluster Explorer", "Search (coming soon)"))

st.sidebar.subheader("Filters (Cluster Explorer)")
city_groups = sorted(df['city_group'].dropna().unique())
city_group = st.sidebar.selectbox("Region (City Group)", ["All"] + city_groups)

if city_group != "All":
    cities = sorted(df[df['city_group'] == city_group]['city'].dropna().unique())
else:
    cities = sorted(df['city'].dropna().unique())
city = st.sidebar.selectbox("City (optional)", ["All"] + list(cities))

cluster_options_df = df.copy()
if city_group != "All":
    cluster_options_df = cluster_options_df[cluster_options_df['city_group'] == city_group]
if city != "All":
    cluster_options_df = cluster_options_df[cluster_options_df['city'] == city]

if 'cluster_label' in df.columns:
    cluster_map = cluster_options_df[['cluster','cluster_label']].drop_duplicates().set_index('cluster')['cluster_label'].to_dict()
    cluster_dropdown = ["All"] + [f"{c} — {cluster_map.get(c,'')}" for c in sorted(cluster_map.keys())]
 
    def parse_cluster_dropdown(s):
        if s=="All": return "All"
          return int(s.split(" — ")[0])
        else:
         cluster_dropdown = ["All"] + [str(c) for c in sorted(cluster_options_df['cluster'].unique())]
		 
    def parse_cluster_dropdown(s):
        return int(s) if s!="All" else "All"

cluster_select = st.sidebar.selectbox("Cluster", cluster_dropdown)

min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), step=0.1)

# ---------------------------
# Filtering function
# ---------------------------
def filter_df(df, city_group, city, cluster_dropdown_val, rating_range):
    tmp = df.copy()
    if city_group != "All":
        tmp = tmp[tmp['city_group'] == city_group]
    if city != "All":
        tmp = tmp[tmp['city'] == city]
    if cluster_dropdown_val != "All":
        cluster_id = parse_cluster_dropdown(cluster_dropdown_val)
        tmp = tmp[tmp['cluster'] == cluster_id]
    tmp = tmp[(tmp['rating'] >= rating_range[0]) & (tmp['rating'] <= rating_range[1])]
    return tmp

filtered = filter_df(df, city_group, city, cluster_select, (min_rating, max_rating))

# ---------------------------
# MAIN: Mode 1 (Cluster Explorer)
# ---------------------------
if mode == "Cluster Explorer":
    st.title("🍽️ Cluster Explorer — Restaurants by Cluster & Region")
    st.markdown(
        "Use Region → Cluster → (optional) City to explore cluster contents. "
        "Top cuisines, rating distribution, top review keywords and a restaurant list are shown."
    )

    # top metrics
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Restaurants Shown", f"{len(filtered):,}")
    c2.metric("Avg rating", f"{filtered['rating'].mean():.2f}" if len(filtered)>0 else "N/A")
    c3.metric("Unique cuisines", f"{len(set([c for row in filtered['cuisine_list'] for c in row])) if len(filtered)>0 else 0}")

    if len(filtered) == 0:
        st.info("No restaurants match this selection. Try a different cluster/region.")
    else:
        # layout: left = charts, right = keywords + list
        left, right = st.columns([2,1])

        # LEFT: cuisine bar + rating histogram
        with left:
            st.subheader("Top Cuisines (selected scope)")
            # flatten cuisine list
            cuisines_flat = [c for row in filtered['cuisine_list'] for c in row]
            if len(cuisines_flat) == 0:
                st.info("No cuisine tags available for this selection.")
            else:
                top_cuisines = pd.Series(cuisines_flat).value_counts().nlargest(12).reset_index()
                top_cuisines.columns = ['cuisine', 'count']
                fig = px.bar(top_cuisines, x='cuisine', y='count', text='count', title="Top Cuisines")
                fig.update_layout(margin=dict(t=40,l=10,r=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Rating distribution")
            fig2 = px.histogram(filtered, x='rating', nbins=10, title="Ratings")
            fig2.update_layout(margin=dict(t=20,l=10,r=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # RIGHT: top keywords + list
        with right:
            st.subheader("Top keywords (reviews)")
            texts = filtered['reviews_preprocessed'].fillna("").astype(str)
            if texts.str.strip().apply(len).sum() == 0:
                st.info("No review texts available for this selection.")
            else:
                cv = CountVectorizer(stop_words='english', max_features=2000)
                Xc = cv.fit_transform(texts)
                counts = np.asarray(Xc.sum(axis=0)).ravel()
                terms = cv.get_feature_names_out()
                top_idx = np.argsort(counts)[::-1][:20]
                top_terms = [(terms[i], int(counts[i])) for i in top_idx]
                kw_df = pd.DataFrame(top_terms, columns=['term','count'])
                fig_kw = px.bar(kw_df, x='term', y='count', text='count', title="Top Review Keywords")
                fig_kw.update_layout(xaxis_tickangle=-45, margin=dict(t=20,l=10,r=10,b=10))
                st.plotly_chart(fig_kw, use_container_width=True)

            st.subheader("Restaurants (top 100 by rating)")
            cols_show = ['restaurant_name','city','cuisine','rating','price_range','ranking','cluster','cluster_label']
            show_df = filtered[cols_show].drop_duplicates().sort_values(by='rating', ascending=False).reset_index(drop=True)
            st.dataframe(show_df.head(100))

# ---------------------------
# Mode 2: Search (placeholder + structure)
# ---------------------------
else:
    st.title("🔎 Search (content-based) — Coming Soon")
    st.markdown(
        "You can type a query (e.g. 'fast food in Amsterdam') and the app will match the query against restaurant review+cuisine TF-IDF vectors "
        "and return top matches. To enable this mode upload the following files to the app folder:\n\n"
        "- tfidf_reviews.pkl  (fitted TfidfVectorizer for reviews)\n- tfidf_cuisine.pkl  (fitted TfidfVectorizer for cuisine tags)\n- X_combined.pkl     (combined sparse matrix aligned with the dataframe rows)\n\n"
        "When these files are present the search box below will become active."
    )
    # UI skeleton for future search mode
    query = st.text_input("Type your search (e.g. 'vegan cafe amsterdam')", "")
    restrict_city_group = st.selectbox("Restrict to region (optional)", ["All"] + city_groups)
    restrict_city = st.selectbox("Restrict to city (optional)", ["All"] + cities)
    st.button("Search (will be active when TF-IDF artifacts are available)")

    # If files present, we could enable logic here (code to do query transform & similarity).
    # We'll leave this as a placeholder for now.
    if os.path.exists(TFIDF_REV_PATH) and os.path.exists(X_COMBINED_PATH):
        st.success("TF-IDF artifacts found — you can enable search mode by uncommenting the code block in the app.")
    else:
        st.info("Artifacts not found. Upload tfidf_reviews.pkl and X_combined.pkl to enable search.")
