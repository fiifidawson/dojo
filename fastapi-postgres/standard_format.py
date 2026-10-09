# ===========================================================================================================
# GET ALL POSTS
@app.get("/posts")
def get_posts(db: Session = Depends(get_db)):

    posts = db.query(models.Post).all() # Query the database for all posts
    return {"data": posts}

# ===========================================================================================================
# GET A POST
@app.get('/posts/{id}')
def get_post(id: int, db: Session = Depends(get_db)):
    
    post = db.query(models.Post).filter(models.Post.id == id).first()
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail= f"Post with id: {id} was not found.")
    return {"post_detail": post}

# ===========================================================================================================
# DELETE A POST
@app.delete("/posts/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(id: int, db: Session = Depends(get_db)):

    post = db.query(models.Post).filter(models.Post.id == id)

    if post.first() == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"post with id: {id} does not exist.")
    post.delete(synchronize_session=False)

    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ===========================================================================================================
# CREATE A POST
@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: Post, db: Session = Depends(get_db)):
    
    new_post = models.Post(**post.model_dump()) # Unpack the post dictionary and create a new post | **: Unpack the dictionary without stating the keys
    db.add(new_post) # After creating the Post you always have to commit it for it to be added to the database
    db.commit() # Commit the post to the database
    db.refresh(new_post) # Refresh the post to get the id of the post
    return {"data": new_post}

# ===========================================================================================================
# UPDATE A POST
@app.put("/posts/{id}")
def update_post(id: int, db: Session = Depends(get_db)):

    post_query = db.query(models.Post).filter(models.Post.id == id)
    
    post = post_query.first()

    if post == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"post with id: {id} does not exist.")
    
    post_query.update(post.model_dump(), synchronize_session=False)

    db.commit()

    return {"data": post_query.first()}