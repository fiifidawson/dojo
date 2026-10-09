### CRUD
Note: posts/users is standard convention. It is standard convention to use plural.

1. Create  - POST      -  /posts      -  @app.post("/posts")

2. Read    - GET       -  /posts/:id  -  @app.get("/posts/{id}") [Getting detailed information on one particular post]

           - GET       -  /posts      -  @app.get("/posts") [Retrieving data from a database]
3. Update  - PUT/PATCH -  /posts/:id  -  @app.put("/posts/{id}") [Updating a pre-existing post]
             PUT - (Pass all the fields to the API in order to update the information)
             PATCH - (Pass the specific field to the API to update the information)

4. Delete   - DELETE   -

<!-- python -m uvicorn main:app --reload -->

uvicorn main:app --reload
uvicorn app.main:app --reload