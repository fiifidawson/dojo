def template(idea: str) -> str:
    template = f"""
    - The 8 AI Skills That Will Separate Winners From Losers in 2025
    - World's Lightest Solid!
    - Why Are 96,000,000 Black Balls on This Reservoir?
    - These 11 income streams made me $220,000 in 2024.
    - I Make $15K/Month With 2 AI Apps
    - Top 5 Reasons Not to Become a Data Analyst
    - What Does a Data Analyst Actually Do?
    - How I Would Become a Data Analyst if I had to Start Over in 2024 | 6 Month Plan
    - How to learn to code FAST using ChatGPT (it's a game changer seriously)
    - 6 Years of Studying Machine Learning in 26 Minutes
    - My honest advice to someone who wants to be a data scientist
    - Complete Python Pandas Data Science Tutorial! (Reading CSV/Excel files, Sorting, Filtering, Groupby)
    - The Complete Machine Learning Roadmap
    - My GPT-evaluator got 1000% better with this simple trick.
    - The 5 paid subscriptions I actually use in 2025 as a Staff Software Engineer
    - AI Explained at 5 Levels of Complexity
    - Docker in 5 Minutes
    - I asked 100 millionaires how to get rich–here's what happened.
    - How I Build Projects (as an AI Engineer)
    - AI Researcher critiques Claude 3.5 sonnet
    - Data scientist explains how to predict the future
    - I bought 10 data science courses so you don’t have to
    - My AI Development Setup (From Scratch)
    - How to Build a Resume Optimizer with AI (Code Walkthrough)
    - I Quit My Job… Here’s How Much I Made 1 Year Later
    - I Was Wrong About AI Consulting (what I learned)

    --
    Given the YouTube video idea write 5 engaging title ideas.

    **Video Idea**: {idea}

    **Additional Guidance**:
    - Titles should be between 30 and 75 characters long
    - Only return the title ideas, nothing else!
    - Title ideas should be written as an ordered markdown list

    """
    return template