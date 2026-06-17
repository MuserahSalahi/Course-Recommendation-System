import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Set page configuration for a professional look
st.set_page_config(page_title="AI Course Recommendation Engine", layout="wide")

# ----------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ----------------------------------------------------
@st.cache_data
def load_data():
    # Load the uploaded Udemy dataset
    df = pd.read_csv("udemy_course_data.csv")
    
    # Fill any missing values in crucial columns
    df['course_title'] = df['course_title'].fillna('')
    df['subject'] = df['subject'].fillna('')
    df['level'] = df['level'].fillna('All Levels')
    
    # Create a combined features column for Content-Based Filtering
    # This aligns with the "Vocabulary Mapping" requirement from the project PDF
    df['content_features'] = df['course_title'] + " " + df['subject'] + " " + df['level']
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'udemy_course_data.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

# ----------------------------------------------------
# 2. AI RECOMMENDATION LOGIC (TF-IDF + COSINE SIMILARITY)
# ----------------------------------------------------
@st.cache_resource
def compute_similarity_matrix(data_frame):
    # Using TF-IDF vectorizer to penalize generic words and reward specific tags
    # As required by the project PDF to move beyond simple binary overlap
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data_frame['content_features'])
    
    # Compute the Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim_matrix = compute_similarity_matrix(df)

def get_recommendations(title, cosine_sim=cosine_sim_matrix, data_frame=df, top_n=5):
    # Get the index of the course that matches the title
    try:
        idx = data_frame[data_frame['course_title'] == title].index[0]
    except IndexError:
        return pd.DataFrame() # Return empty if course not found

    # Get the pairwise similarity scores of all courses with that course
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the courses based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-N most similar courses (excluding itself)
    sim_scores = sim_scores[1:top_n+1]

    # Get the course indices
    course_indices = [i[0] for i in sim_scores]

    # Return the top N most similar courses with relevant columns
    return data_frame.iloc[course_indices][['course_title', 'subject', 'level', 'price', 'num_subscribers', 'url']]

# ----------------------------------------------------
# 3. STREAMLIT USER INTERFACE (UI)
# ----------------------------------------------------
st.title(" AI-Powered Course Recommendation System")
st.markdown("This engine maps user choices to item features using **TF-IDF Vectorization** and **Cosine Similarity Math**.")

# Sidebar for Navigation and User Inputs
st.sidebar.header("User Preference Panel")

# Step 1: Let the user pick a course they are interested in
all_courses = df['course_title'].values
selected_course = st.sidebar.selectbox("Select a Course You Like:", all_courses)

# Step 2: Select number of recommendations
num_recommendations = st.sidebar.slider("Number of Recommendations:", min_value=3, max_value=10, value=5)

# Main Dashboard Layout tabs
tab1, tab2 = st.tabs([" Recommendations Engine", " Dataset Analytics & Graphs"])

with tab1:
    st.header("Tailored Recommendations For You")
    st.write(f"Based on your interest in: **{selected_course}**")
    
    if selected_course:
        # Generate recommendations using the Content-Based Logic
        recommendations = get_recommendations(selected_course, top_n=num_recommendations)
        
        if not recommendations.empty:
            # Display recommendations beautifully in loops/cards
            for index, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"###  {row['course_title']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Subject:** {row['subject']}")
                    with col2:
                        st.write(f"**Level:** {row['level']}")
                    with col3:
                        st.write(f"**Price:** ${row['price'] if row['price'] != 'Free' else 'Free'}")

                    st.write(f"👥 **Subscribers:** {row['num_subscribers']:,}")
                    st.markdown(f"[Go to Course Link]({row['url']})")
                    st.markdown("---")
        else:
            st.warning("No recommendations could be generated for this selection.")

with tab2:
    st.header("Dataset Insights & Analytical Graphs")
    st.markdown("These charts highlight the structural distribution of features across the items repository.")
    
    # Create two columns for side-by-side graphs
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        # Graph 1: Distribution of courses across different subjects
        st.subheader("Course Distribution by Subject")
        subject_counts = df['subject'].value_counts().reset_index()
        subject_counts.columns = ['Subject', 'Count']
        fig_pie = px.pie(subject_counts, values='Count', names='Subject', 
                         title="Market Share of Course Subjects",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    with g_col2:
        # Graph 2: Experience Levels distribution
        st.subheader("Courses Availability by Target Level")
        level_counts = df['level'].value_counts().reset_index()
        level_counts.columns = ['Level', 'Count']
        fig_bar = px.bar(level_counts, x='Level', y='Count', 
                         title="Number of Courses per Experience Level",
                         labels={'Count': 'Number of Courses'},
                         color='Level', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Graph 3: Top 10 Most Popular Courses in the Dataset
    st.subheader(" Top 10 Most Popular Courses by Subscriber Count")
    top_courses = df.nlargest(10, 'num_subscribers')[['course_title', 'num_subscribers', 'subject']]
    fig_pop = px.bar(top_courses, x='num_subscribers', y='course_title', 
                     orientation='h', color='subject',
                     title="Most Subscribed Courses",
                     labels={'num_subscribers': 'Total Subscribers', 'course_title': 'Course Title'},
                     category_orders={"course_title": top_courses['course_title'].values[::-1]})
    st.plotly_chart(fig_pop, use_container_width=True)




