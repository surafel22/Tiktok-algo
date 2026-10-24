import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="TikTok Rec Engine", layout="wide")
st.title("🚀 Advanced Hybrid Video Recommendation Engine")
st.write("This app uses Collaborative Filtering and Content-Based Filtering to generate a personalized 'For You' feed.")

@st.cache_data
def load_data():
    np.random.seed(42)
    n_users, n_videos, n_interactions = 500, 2000, 15000
    categories = ['Comedy', 'Tech', 'Gaming', 'Music', 'Sports', 'Education', 'Lifestyle']
    video_cats = {vid: np.random.choice(categories) for vid in range(1001, 1001+n_videos)}
    video_lens = {vid: np.random.choice([15, 30, 45, 60, 120]) for vid in range(1001, 1001+n_videos)}

    data = {
        'User_ID': np.random.randint(1, n_users+1, n_interactions),
        'Video_ID': np.random.randint(1001, 1001+n_videos, n_interactions),
        'Watch_Time_Sec': np.random.exponential(scale=25, size=n_interactions),
        'Liked': np.random.choice([1, 0], n_interactions, p=[0.15, 0.85]),
        'Shared': np.random.choice([1, 0], n_interactions, p=[0.05, 0.95]),
        'Hour': np.random.randint(0, 24, n_interactions)
    }
    df = pd.DataFrame(data)
    df['Category'] = df['Video_ID'].map(video_cats)
    df['Video_Length_Sec'] = df['Video_ID'].map(video_lens)

    df['Watch_Ratio'] = (df['Watch_Time_Sec'] / df['Video_Length_Sec']).clip(upper=1.5)
    df['Is_Skipped'] = (df['Watch_Time_Sec'] < 3.0).astype(int)
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)

    df['Engagement_Score'] = (df['Watch_Ratio'] * 2.0) + (df['Liked'] * 1.5) + (df['Shared'] * 2.0) - (df['Is_Skipped'] * 1.0)
    scaler = MinMaxScaler(feature_range=(1, 5))
    df['Implicit_Rating'] = scaler.fit_transform(df[['Engagement_Score']])
    
    return df, video_cats

@st.cache_resource
def train_model(df):
    user_video_matrix = df.pivot_table(index='User_ID', columns='Video_ID', values='Implicit_Rating').fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    latent_matrix = svd.fit_transform(user_video_matrix)
    predicted_ratings = np.dot(latent_matrix, svd.components_)
    cf_preds_df = pd.DataFrame(predicted_ratings, columns=user_video_matrix.columns, index=user_video_matrix.index)
    return cf_preds_df, latent_matrix, user_video_matrix

df, video_cats = load_data()
cf_preds_df, latent_matrix, user_video_matrix = train_model(df)

def generate_for_you_feed(user_id, top_n=10):
    if user_id not in cf_preds_df.index:
        return pd.DataFrame()
        
    user_preds = cf_preds_df.loc[user_id].reset_index()
    user_preds.columns = ['Video_ID', 'CF_Score']
    
    seen_videos = df[df['User_ID'] == user_id]['Video_ID'].unique()
    candidates = user_preds[~user_preds['Video_ID'].isin(seen_videos)].copy()
    
    user_history = df[df['User_ID'] == user_id]
    cat_affinity = user_history.groupby('Category')['Implicit_Rating'].mean().to_dict()
    
    candidates['Category'] = candidates['Video_ID'].map(video_cats)
    candidates['Affinity_Boost'] = candidates['Category'].map(cat_affinity).fillna(0) * 0.5
    candidates['Final_Rec_Score'] = candidates['CF_Score'] + candidates['Affinity_Boost']
    
    return candidates.sort_values(by='Final_Rec_Score', ascending=False).head(top_n)

st.sidebar.header("🎯 Feed Generator Settings")
target_user_id = st.sidebar.slider("Select User ID", min_value=1, max_value=500, value=42)
feed_size = st.sidebar.slider("Number of Videos", min_value=5, max_value=20, value=10)

st.subheader(f"📱 Personalized Feed for User #{target_user_id}")
feed = generate_for_you_feed(target_user_id, top_n=feed_size)

if not feed.empty:
    st.dataframe(feed[['Video_ID', 'Category', 'CF_Score', 'Affinity_Boost', 'Final_Rec_Score']], use_container_width=True)
else:
    st.warning("User not found. Needs more data to resolve cold-start.")

st.subheader("🌌 3D Visualization of User Behavioral Embeddings")
embeddings = pd.DataFrame(latent_matrix[:100, :3], columns=['Latent_1', 'Latent_2', 'Latent_3'])
embeddings['User_ID'] = user_video_matrix.index[:100].astype(str)

fig = px.scatter_3d(
    embeddings, x='Latent_1', y='Latent_2', z='Latent_3',
    color='Latent_1', hover_name='User_ID',
    color_continuous_scale='Inferno'
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig, use_container_width=True)