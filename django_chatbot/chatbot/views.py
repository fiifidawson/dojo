from django.shortcuts import render
from django.http import JsonResponse
import openai
from openai import OpenAI
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API key from environment variable or direct key
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def ask_openai(message):
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {"role": "user", 
             "content": message}
        ],
    )
    
    print(response)
    # Extract the response text
    answer = response.choices[0].message.reasoning
    return answer


# Create your views here.
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')