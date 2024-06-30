from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import difflib

# Load the data and similarity matrix
movies_data = pd.read_csv('movies_data.csv')
similarity = joblib.load('similarity_matrix.pkl')

app = FastAPI()

class MovieRequest(BaseModel):
    movie_name: str

@app.post('/recommend')
async def recommend(request: MovieRequest):
    movie_name = request.movie_name
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[:30]):
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    
    return {'recommendations': recommended_movies}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
