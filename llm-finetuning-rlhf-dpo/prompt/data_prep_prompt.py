def template(idea: str) -> str:

    template = f"""Given the YouTube video idea write an engaging title.

    **Video Idea**: {idea}

    **Additional Guidance**:
    - Title should be between 30 and 75 characters long
    - Only return the title idea, nothing else!"""
    return template