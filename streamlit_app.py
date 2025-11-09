import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import numpy as np

# ================================================================
# SETUP
# ================================================================
st.set_page_config(
    page_title="sample_mflix Cinema Dashboard",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better look
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #00d4ff;
        padding: 1rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# CONNECT TO DATABASE  (SAFE for local .env and Streamlit Cloud)
# ================================================================

@st.cache_resource
def get_database():
    """Try Streamlit Cloud secrets first; if unavailable, load from local .env."""
    MONGO_URI = None
    DB_NAME = None

    # 1) Try Streamlit Secrets (used on Streamlit Cloud)
    try:
        # Accessing st.secrets can raise if no secrets.toml exists locally
        MONGO_URI = st.secrets["MONGO_URI"] if "MONGO_URI" in st.secrets else None
        DB_NAME = st.secrets["DB_NAME"] if "DB_NAME" in st.secrets else None
    except Exception:
        # No secrets.toml locally — that's fine, we'll fall back to .env
        MONGO_URI = None
        DB_NAME = None

    # 2) Fallback to local .env when running on your laptop
    if not MONGO_URI:
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_URI")
        DB_NAME = DB_NAME or os.getenv("DB_NAME", "sample_mflix")

    if not MONGO_URI:
        st.error("Please set MONGO_URI via Streamlit Secrets (cloud) or a local .env file.")
        st.stop()

    client = MongoClient(MONGO_URI)
    return client[DB_NAME or "sample_mflix"]

db = get_database()

# ================================================================
# LOAD DATA (CACHED)
# ================================================================

@st.cache_data(ttl=600)
def load_all_data():
    """Load all necessary data at once"""
    # Basic stats
    stats = {
        "movies": db.movies.count_documents({}),
        "comments": db.comments.count_documents({}),
        "users": db.users.count_documents({})
    }

    # Movies with ratings
    movies_cursor = db.movies.find(
        {"imdb.rating": {"$exists": True}, "year": {"$gte": 1990}},
        {"title": 1, "year": 1, "genres": 1, "imdb.rating": 1, "imdb.votes": 1}
    ).limit(5000)

    movies_list = []
    for movie in movies_cursor:
        movies_list.append({
            "title": movie.get("title"),
            "year": movie.get("year"),
            "rating": (movie.get("imdb") or {}).get("rating"),
            "votes": (movie.get("imdb") or {}).get("votes", 0),
            "genres": movie.get("genres", [])
        })

    df_movies = pd.DataFrame(movies_list)

    # Genre analysis
    genre_pipeline = [
        {"$unwind": "$genres"},
        {"$group": {
            "_id": "$genres",
            "count": {"$sum": 1},
            "avg_rating": {"$avg": "$imdb.rating"}
        }},
        {"$match": {"count": {"$gte": 100}}},
        {"$sort": {"count": -1}}
    ]
    df_genres = pd.DataFrame(list(db.movies.aggregate(genre_pipeline)))
    if not df_genres.empty:
        df_genres = df_genres.rename(columns={"_id": "genre"})

    # Yearly trends
    yearly_pipeline = [
        {"$match": {"year": {"$gte": 1990}, "imdb.rating": {"$exists": True}}},
        {"$group": {
            "_id": "$year",
            "avg_rating": {"$avg": "$imdb.rating"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    df_yearly = pd.DataFrame(list(db.movies.aggregate(yearly_pipeline)))
    if not df_yearly.empty:
        df_yearly = df_yearly.rename(columns={"_id": "year"})

    return stats, df_movies, df_genres, df_yearly

# Load everything
stats, df_movies, df_genres, df_yearly = load_all_data()

# ================================================================
# HEADER
# ================================================================
st.markdown('<div class="big-title"> Cinema Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #888;'>Connected to <b>{db.name}</b> on Azure Cloud </p>", unsafe_allow_html=True)
st.markdown("---")

# ================================================================
# SIDEBAR FILTERS
# ================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/movie.png", width=80)
    st.title(" Filters")

    st.success(" Connected to Cloud")
    st.info(f" {stats['movies']:,} movies\n\n {stats['comments']:,} comments")

    st.markdown("---")

    # Year filter
    year_min, year_max = st.slider("Year Range", 1990, 2023, (2000, 2023))

    # Rating filter
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 5.0, 0.5)

    # Genre filter
    all_genres = sorted(df_genres['genre'].tolist()) if not df_genres.empty else []
    selected_genres = st.multiselect(
        "Select Genres",
        all_genres,
        default=all_genres[:3] if all_genres else []
    )

# Apply filters
df_filtered = df_movies[
    (df_movies['year'] >= year_min) &
    (df_movies['year'] <= year_max) &
    (df_movies['rating'] >= min_rating)
].copy()

if selected_genres:
    df_filtered = df_filtered[
        df_filtered['genres'].apply(lambda x: any(g in selected_genres for g in x) if isinstance(x, list) else False)
    ].copy()

# ================================================================
# MAIN TABS
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Trends", " Genres", " Top Movies"])

# ----------------------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------------------
with tab1:
    st.subheader(" Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h2>{len(df_filtered):,}</h2>
            <p>Total Movies</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_rating = df_filtered['rating'].mean()
        st.markdown(f"""
        <div class='metric-box'>
            <h2>{avg_rating:.2f}</h2>
            <p>Avg Rating</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_votes = df_filtered['votes'].sum()
        st.markdown(f"""
        <div class='metric-box'>
            <h2>{total_votes/1e6:.1f}M</h2>
            <p>Total Votes</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        top_year = df_filtered['year'].mode()[0] if not df_filtered.empty else 2020
        st.markdown(f"""
        <div class='metric-box'>
            <h2>{int(top_year)}</h2>
            <p>Peak Year</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Rating distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Rating Distribution")
        fig = px.histogram(
            df_filtered,
            x='rating',
            nbins=30,
            title="How Movies Are Rated",
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(" Movies by Year")
        yearly_count = df_filtered.groupby('year').size().reset_index(name='count')
        fig = px.bar(
            yearly_count,
            x='year',
            y='count',
            title="Production Volume Over Time",
            color_discrete_sequence=['#ff00ff']
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Quick insights
    st.subheader(" Quick Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **Highest Rated:**  
        {df_filtered.nlargest(1, 'rating')['title'].values[0] if not df_filtered.empty else 'N/A'}  
        Rating: {df_filtered['rating'].max():.2f} 
        """)

    with col2:
        most_voted = df_filtered.nlargest(1, 'votes')
        st.success(f"""
        **Most Popular:**  
        {most_voted['title'].values[0] if not most_voted.empty else 'N/A'}  
        Votes: {most_voted['votes'].values[0]:,.0f} 👥
        """)

    with col3:
        recent_avg = df_filtered[df_filtered['year'] >= 2020]['rating'].mean()
        st.warning(f"""
        **Recent Quality:**  
        2020+ Average: {recent_avg:.2f}  
        vs Overall: {avg_rating:.2f}
        """)

# ----------------------------------------------------------------
# TAB 2: TRENDS
# ----------------------------------------------------------------
with tab2:
    st.subheader(" Rating Trends Over Time")

    if not df_yearly.empty:
        # Ensure correct columns
        df_yearly = df_yearly.rename(columns={"_id": "year"}) if "_id" in df_yearly.columns else df_yearly

        # Filter yearly data
        df_yearly_filtered = df_yearly[
            (df_yearly['year'] >= year_min) &
            (df_yearly['year'] <= year_max)
        ]

        # Create dual-axis chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_yearly_filtered['year'],
            y=df_yearly_filtered['avg_rating'],
            name='Average Rating',
            line=dict(color='#00d4ff', width=3),
            yaxis='y'
        ))

        fig.add_trace(go.Bar(
            x=df_yearly_filtered['year'],
            y=df_yearly_filtered['count'],
            name='Number of Movies',
            marker_color='rgba(255, 0, 255, 0.5)',
            yaxis='y2'
        ))

        fig.update_layout(
            title="Quality vs Volume: The Cinema Evolution",
            xaxis_title="Year",
            yaxis=dict(title="Average Rating", side='left'),
            yaxis2=dict(title="Movie Count", overlaying='y', side='right'),
            template='plotly_dark',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Trend analysis
        col1, col2 = st.columns(2)

        with col1:
            z = np.polyfit(df_yearly_filtered['year'], df_yearly_filtered['avg_rating'], 1)
            trend = " Improving" if z[0] > 0 else " Declining"

            st.markdown(f"""
            ###  Trend Analysis

            **Overall Direction:** {trend}  
            **Change Rate:** {abs(z[0]):.4f} points/year  
            **Peak Year:** {int(df_yearly_filtered.loc[df_yearly_filtered['avg_rating'].idxmax(), 'year'])}  
            **Peak Rating:** {df_yearly_filtered['avg_rating'].max():.2f}/10

            ---

            **Interpretation:**  
            Cinema quality has {'improved' if z[0] > 0 else 'declined'} over the selected period. 
            This reflects {'rising' if z[0] > 0 else 'changing'} audience expectations and 
            {'better' if z[0] > 0 else 'more'} production standards.
            """)

        with col2:
            # Show yearly data table
            st.markdown("###  Recent Years Data")
            df_display = df_yearly_filtered.tail(10).copy()
            df_display['avg_rating'] = df_display['avg_rating'].round(2)
            df_display.columns = ['Year', 'Avg Rating', 'Movies']
            st.dataframe(
                df_display.sort_values('Year', ascending=False),
                hide_index=True,
                use_container_width=True,
                height=350
            )

# ----------------------------------------------------------------
# TAB 3: GENRES
# ----------------------------------------------------------------
with tab3:
    st.subheader(" Genre Analysis")

    if not df_genres.empty:
        df_genres = df_genres.rename(columns={"_id": "genre"}) if "_id" in df_genres.columns else df_genres

        col1, col2 = st.columns(2)

        with col1:
            # Top genres by count
            top_genres = df_genres.nlargest(10, 'count')
            fig = px.bar(
                top_genres,
                y='genre',
                x='count',
                orientation='h',
                title="Top 10 Genres by Volume",
                color='count',
                color_continuous_scale='Viridis',
                text='count'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(
                template='plotly_dark',
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top genres by rating
            top_quality = df_genres.nlargest(10, 'avg_rating')
            fig = px.bar(
                top_quality,
                y='genre',
                x='avg_rating',
                orientation='h',
                title="Top 10 Genres by Quality",
                color='avg_rating',
                color_continuous_scale='RdYlGn',
                text='avg_rating'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                template='plotly_dark',
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot: Volume vs Quality
        st.subheader(" Volume vs Quality Analysis")

        fig = px.scatter(
            df_genres,
            x='count',
            y='avg_rating',
            size='count',
            color='avg_rating',
            hover_data=['genre'],
            title="Genre Performance: Popularity vs Quality",
            labels={'count': 'Number of Movies', 'avg_rating': 'Average Rating'},
            color_continuous_scale='Turbo'
        )

        # Annotate top genres
        for _, row in df_genres.head(5).iterrows():
            fig.add_annotation(
                x=row['count'],
                y=row['avg_rating'],
                text=row['genre'],
                showarrow=True,
                arrowhead=2,
                arrowcolor='white'
            )

        fig.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Business insight
        st.markdown("""
        ###  Business Insight

        **The Quality-Volume Paradox:**
        - Niche genres (Documentary, Film-Noir) show **higher ratings** but **lower volume**
        - Mass-market genres (Drama, Comedy) dominate production but have **average quality**
        - **Recommendation:** Invest 30% of budget in underserved high-quality genres
        """)

        # Full genre table
        st.subheader(" Complete Genre Database")
        df_genre_display = df_genres[['genre', 'count', 'avg_rating']].copy()
        df_genre_display.columns = ['Genre', 'Movies', 'Avg Rating']
        df_genre_display['Avg Rating'] = df_genre_display['Avg Rating'].round(2)

        st.dataframe(
            df_genre_display,
            hide_index=True,
            use_container_width=True,
            height=400
        )

# ----------------------------------------------------------------
# TAB 4: TOP MOVIES
# ----------------------------------------------------------------
with tab4:
    st.subheader(" Top Performing Movies")

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ["Rating", "Votes", "Both (Balanced)"]
        )

    with col2:
        show_count = st.slider("Show Top", 10, 50, 20)

    # Calculate scores (avoid SettingWithCopy warnings)
    df_work = df_filtered.copy()
    if sort_by == "Rating":
        df_sorted = df_work.nlargest(show_count, 'rating')
    elif sort_by == "Votes":
        df_sorted = df_work.nlargest(show_count, 'votes')
    else:
        df_work['score'] = df_work['rating'] * np.log10(df_work['votes'] + 1)
        df_sorted = df_work.nlargest(show_count, 'score')

    # Visualization
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.bar(
            df_sorted.head(15),
            y='title',
            x='rating',
            orientation='h',
            title=f"Top 15 Movies by {sort_by}",
            color='rating',
            color_continuous_scale='Viridis',
            text='rating'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            template='plotly_dark',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("###  Statistics")

        st.metric("Highest Rating", f"{df_sorted['rating'].max():.2f} ")
        st.metric("Most Votes", f"{df_sorted['votes'].max():,.0f} ")
        st.metric("Avg Year", f"{int(df_sorted['year'].mean())}")

        st.markdown("---")
        st.markdown("###  Top Movie")

        top_movie = df_sorted.iloc[0]
        st.markdown(f"""
        **{top_movie['title']}**  
        Year: {int(top_movie['year'])}  
        Rating: {top_movie['rating']:.2f}   
        Votes: {top_movie['votes']:,.0f} 
        """)

    # Full table
    st.subheader(" Complete Rankings")

    df_display = df_sorted[['title', 'year', 'rating', 'votes', 'genres']].copy()
    df_display['genres'] = df_display['genres'].apply(lambda x: ', '.join(x[:3]) if isinstance(x, list) else 'N/A')
    df_display.columns = ['Title', 'Year', 'Rating', 'Votes', 'Genres']

    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rating": st.column_config.NumberColumn(format="%.2f "),
            "Votes": st.column_config.NumberColumn(format="%d"),
        },
        height=400
    )

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("Data Source: Azure Cosmos DB")
with col2:
    st.markdown(" Built with: Streamlit + MongoDB")
with col3:
    st.markdown(" Last Updated: Live")

st.markdown("""
<p style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
© 2025 Cinema Analytics Dashboard | DSA508 Big Data Project | Built by Sowmya Galanki
</p>
""", unsafe_allow_html=True)
