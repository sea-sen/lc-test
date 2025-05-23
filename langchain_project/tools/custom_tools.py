from langchain.tools import tool
from pydantic import BaseModel, Field
import random

# --- Simple Calculator Tool ---
class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

@tool("simple_calculator", args_schema=CalculatorInput)
def simple_calculator(a: float, b: float) -> dict:
    """
    A simple calculator that can add, subtract, multiply, and divide two numbers.
    Returns a dictionary with the operation and result.
    """
    return {
        "addition": a + b,
        "subtraction": a - b,
        "multiplication": a * b,
        "division": a / b if b != 0 else "undefined (division by zero)"
    }

# --- Mock Weather Tool ---
class WeatherInput(BaseModel):
    location: str = Field(description="The city or region for which to get the weather.")

@tool("get_mock_weather", args_schema=WeatherInput)
def get_mock_weather(location: str) -> dict:
    """
    A mock weather service that returns a fake weather forecast for a given location.
    """
    conditions = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy"]
    temperature = random.randint(-10, 35) # Celsius
    humidity = random.randint(30, 90) # Percent

    return {
        "location": location,
        "condition": random.choice(conditions),
        "temperature_celsius": temperature,
        "humidity_percent": humidity,
        "forecast": f"The weather in {location} is {random.choice(conditions)}, with a temperature of {temperature}°C and humidity of {humidity}%."
    }

if __name__ == '__main__':
    print("Testing custom tools...")
    
    # Test Calculator
    calc_result = simple_calculator.invoke({"a": 10, "b": 5})
    print(f"\nCalculator (10, 5): {calc_result}")
    calc_result_zero_div = simple_calculator.invoke({"a": 10, "b": 0})
    print(f"Calculator (10, 0): {calc_result_zero_div}")

    # Test Weather
    weather_sf = get_mock_weather.invoke({"location": "San Francisco"})
    print(f"\nWeather (San Francisco): {weather_sf}")
    weather_london = get_mock_weather.invoke({"location": "London"})
    print(f"\nWeather (London): {weather_london}")
