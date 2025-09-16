import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import ast, re, os
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy.sparse import hstack

st.set_page_config(
    page_title="Restaurant Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

DF_PATH = "restaurant_reviews_clustered.pkl"
TFIDF_REV_PATH = "X_reviews.pkl"
TFIDF_CUIS_PATH = "X_Cuisine.pkl"
DF_PATH2 = "df_cleaned_2.pkl"
cuisnie_reviews_combined = "X_combined.pkl"


def parse_cuisine_field(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                return [str(i).strip() for i in lst if str(i).strip()]
        except Exception:
            pass
    if re.search(r"[;,/|]", s):
        parts = re.split(r"[;,/|]+", s)
        return [p.strip() for p in parts if p.strip()]
    if " & " in s or " and " in s:
        parts = re.split(r"\s+&\s+|\s+and\s+", s)
        return [p.strip() for p in parts if p.strip()]
    return [s]


# ---------------------------
# load dataframe
# ---------------------------
@st.cache_data(show_spinner=True)
def load_df(path=DF_PATH, path2=DF_PATH2):
    if not os.path.exists(path):
        st.error(f"Data file not found at {path}. Place restaurant_reviews_clustered.pkl in the app folder.")
        return pd.DataFrame()

    # Load main dataframe
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        st.error("Unsupported file format. Only .pkl supported right now.")
        return pd.DataFrame()

    # Attach reviews_preprocessed if second file exists
    if os.path.exists(path2):
        df_reviews = pd.read_pickle(path2)
        if "reviews_preprocessed" in df_reviews.columns:
            if len(df) == len(df_reviews):
                df["reviews_preprocessed"] = df_reviews["reviews_preprocessed"].values
            else:
                st.warning(
                    f"Row mismatch: {len(df)} rows in {path}, but {len(df_reviews)} rows in {path2}. "
                    "reviews_preprocessed not added."
                )
        else:
            st.warning(f"No column 'reviews_preprocessed' found in {path2}.")
    else:
        st.info(f"Optional file {path2} not found. reviews_preprocessed will be empty.")
        df["reviews_preprocessed"] = ""

    return df

def load_tfidf_artifacts():
    if not (os.path.exists(TFIDF_REV_PATH) and os.path.exists(TFIDF_CUIS_PATH) and os.path.exists(cuisnie_reviews_combined)):
        return None, None, None
    tfidf_rev = joblib.load(TFIDF_REV_PATH)
    tfidf_cuis = joblib.load(TFIDF_CUIS_PATH)
    X_combined = joblib.load(cuisnie_reviews_combined)
    return tfidf_rev, tfidf_cuis, X_combined



df = load_df()
if df.empty:
    st.stop()

# pre-parse cuisine lists (cached)
@st.cache_data
def get_cuisine_lists(series):
    return series.apply(parse_cuisine_field)


df["cuisine_list"] = get_cuisine_lists(df["cuisine"])

st.sidebar.header("Mode & Filters")
mode = st.sidebar.radio("Mode", ("Cluster Explorer", "Search (coming soon)"))

st.sidebar.subheader("Filters (Cluster Explorer)")
city_groups = sorted(df["city_group"].dropna().unique())
city_group = st.sidebar.selectbox("Region (City Group)", ["All"] + city_groups)

if city_group != "All":
    cities = sorted(df[df["city_group"] == city_group]["city"].dropna().unique())
else:
    cities = sorted(df["city"].dropna().unique())
city = st.sidebar.selectbox("City (optional)", ["All"] + list(cities))

cluster_options_df = df.copy()
if city_group != "All":
    cluster_options_df = cluster_options_df[cluster_options_df["city_group"] == city_group]
if city != "All":
    cluster_options_df = cluster_options_df[cluster_options_df["city"] == city]

if "cluster_label" in df.columns:
    cluster_map = (
        cluster_options_df[["cluster", "cluster_label"]]
        .drop_duplicates()
        .set_index("cluster")["cluster_label"]
        .to_dict()
    )
    cluster_dropdown = ["All"] + [f"{c} — {cluster_map.get(c, '')}" for c in sorted(cluster_map.keys())]

    def parse_cluster_dropdown(s):
        return int(s.split(" — ")[0]) if s != "All" else "All"

else:
    cluster_dropdown = ["All"] + [str(c) for c in sorted(cluster_options_df["cluster"].unique())]

    def parse_cluster_dropdown(s):
        return int(s) if s != "All" else "All"

cluster_select = st.sidebar.selectbox("Cluster", cluster_dropdown)

min_rating, max_rating = st.sidebar.slider("Rating range", 0.0, 5.0, (0.0, 5.0), step=0.1)


# ---------------------------
# Filtering function
# ---------------------------
def filter_df(df, city_group, city, cluster_dropdown_val, rating_range):
    tmp = df.copy()
    if city_group != "All":
        tmp = tmp[tmp["city_group"] == city_group]
    if city != "All":
        tmp = tmp[tmp["city"] == city]
    if cluster_dropdown_val != "All":
        cluster_id = parse_cluster_dropdown(cluster_dropdown_val)
        tmp = tmp[tmp["cluster"] == cluster_id]
    tmp = tmp[(tmp["rating"] >= rating_range[0]) & (tmp["rating"] <= rating_range[1])]
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
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Restaurants Shown", f"{len(filtered):,}")
    c2.metric("Avg rating", f"{filtered['rating'].mean():.2f}" if len(filtered) > 0 else "N/A")
    c3.metric(
        "Unique cuisines",
        f"{len(set([c for row in filtered['cuisine_list'] for c in row])) if len(filtered) > 0 else 0}",
    )

    if len(filtered) == 0:
        st.info("No restaurants match this selection. Try a different cluster/region.")
    else:
        # layout: left = charts, right = keywords + list
        left, right = st.columns([2, 1])

        # LEFT: cuisine bar + rating histogram
        with left:
            st.subheader("Top Cuisines (selected scope)")
            cuisines_flat = [c for row in filtered["cuisine_list"] for c in row]
            if len(cuisines_flat) == 0:
                st.info("No cuisine tags available for this selection.")
            else:
                top_cuisines = (
                    pd.Series(cuisines_flat).value_counts().nlargest(12).reset_index()
                )
                top_cuisines.columns = ["cuisine", "count"]
                fig = px.bar(top_cuisines, x="cuisine", y="count", text="count", title="Top Cuisines")
                fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Rating distribution")
            fig2 = px.histogram(filtered, x="rating", nbins=10, title="Ratings")
            fig2.update_layout(margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        # RIGHT: top keywords + list
        with right:
            st.subheader("Top keywords (reviews)")
            texts = filtered["reviews_preprocessed"].fillna("").astype(str)
            if texts.str.strip().apply(len).sum() == 0:
                st.info("No review texts available for this selection.")
            else:
                cv = CountVectorizer(stop_words="english", max_features=2000)
                Xc = cv.fit_transform(texts)
                counts = np.asarray(Xc.sum(axis=0)).ravel()
                terms = cv.get_feature_names_out()
                top_idx = np.argsort(counts)[::-1][:20]
                top_terms = [(terms[i], int(counts[i])) for i in top_idx]
                kw_df = pd.DataFrame(top_terms, columns=["term", "count"])
                fig_kw = px.bar(kw_df, x="term", y="count", text="count", title="Top Review Keywords")
                fig_kw.update_layout(xaxis_tickangle=-45, margin=dict(t=20, l=10, r=10, b=10))
                st.plotly_chart(fig_kw, use_container_width=True)

            st.subheader("Restaurants (top 100 by rating)")
            cols_show = [
                "restaurant_name",
                "city",
                "cuisine",
                "rating",
                "price_range",
                "ranking",
                "cluster",
                "cluster_label",
            ]
            show_df = (
                filtered[cols_show]
                .drop_duplicates()
                .sort_values(by="rating", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(show_df.head(100))

# ---------------------------
# Mode 2: Search (placeholder + structure)
# ---------------------------
else:
    st.title("🔎 Search (content-based)")
    query = st.text_input("Type your search (e.g. 'vegan cafe amsterdam')", "")

    # make city_group + city mandatory
    city_group = st.selectbox("Region (City Group)", city_groups)
    cities = sorted(df[df["city_group"] == city_group]["city"].dropna().unique())
    city = st.selectbox("City", cities)

    tfidf_rev, tfidf_cuis, X_combined = load_tfidf_artifacts()

    if tfidf_rev is None:
        st.error("Missing TF-IDF artifacts. Please upload tfidf_reviews.pkl, tfidf_cuisine.pkl, and X_combined.pkl")
    else:
        if query:
            # transform query into combined vector
            q_rev = tfidf_rev.transform([query])
            q_cuis = tfidf_cuis.transform([query])
            q_vec = hstack([q_rev, q_cuis])

            # restrict to selected city_group + city
            mask = (df["city_group"] == city_group) & (df["city"] == city)
            subset_df = df[mask].reset_index(drop=True)
            X_subset = X_combined[mask]

            if len(subset_df) == 0:
                st.warning("No restaurants found for this selection.")
            else:
                # cosine similarity
                sims = cosine_similarity(q_vec, X_subset).ravel()
                subset_df["similarity"] = sims

                # show top 20
                top_results = subset_df.sort_values("similarity", ascending=False).head(20)
                st.dataframe(
                    top_results[
                        ["restaurant_name", "cuisine", "rating", "cluster", "similarity"]
                    ]
                )