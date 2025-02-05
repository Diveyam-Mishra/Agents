import requests
from django.shortcuts import render
from django.http import JsonResponse

def weather_view(request):
    API_KEY = "your_openweathermap_api_key"  # Replace with your actual API key
    if request.method == "POST":
        city = request.POST.get("city")
        api_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"],
            }
            return render(request, "weather_app/index.html", {"weather": weather_data})
        else:
            return render(request, "weather_app/index.html", {"error": "City not found."})
    return render(request, "weather_app/index.html")