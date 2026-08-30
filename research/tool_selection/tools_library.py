"""
research/tool_selection/tools_library.py
========================================
50 deterministic, mock tool definitions for the tool-selection study.

All callables are pure-Python with no external API calls (no API keys,
no network I/O).  Outputs are deterministic and reproducible.

Domains covered (10 domains, ~5 tools each):
  1.  Math & arithmetic
  2.  Unit conversion — length
  3.  Unit conversion — weight/mass      ← deliberately confusable with #2
  4.  Unit conversion — temperature
  5.  Date & time
  6.  Currency conversion (mocked rates, 2024-01-01 snapshot)
  7.  String manipulation
  8.  Statistics & probability
  9.  Geocoding & geography (mocked)
  10. General knowledge / trivia (mocked)

Confusable pairs (intentionally similar descriptions):
  - convert_km_to_miles  vs  convert_miles_to_km
  - convert_kg_to_lbs   vs  convert_lbs_to_kg
  - celsius_to_fahrenheit vs fahrenheit_to_celsius
  - get_city_population  vs  get_country_population
  - word_count          vs  character_count
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import List

from lunaite.tools import Tool, ToolRegistry


# ═══════════════════════════════════════════════════════════════════════════ #
# 1. Math & arithmetic                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def _add(a: float, b: float) -> float:
    return a + b

def _subtract(a: float, b: float) -> float:
    return a - b

def _multiply(a: float, b: float) -> float:
    return a * b

def _divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

def _power(base: float, exponent: float) -> float:
    return base ** exponent

MATH_TOOLS = [
    Tool(
        name="add_numbers",
        description="Add two numbers together and return the sum.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
        callable=_add,
    ),
    Tool(
        name="subtract_numbers",
        description="Subtract the second number from the first and return the difference.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Number to subtract from"},
                "b": {"type": "number", "description": "Number to subtract"},
            },
            "required": ["a", "b"],
        },
        callable=_subtract,
    ),
    Tool(
        name="multiply_numbers",
        description="Multiply two numbers together and return the product.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First factor"},
                "b": {"type": "number", "description": "Second factor"},
            },
            "required": ["a", "b"],
        },
        callable=_multiply,
    ),
    Tool(
        name="divide_numbers",
        description="Divide the first number by the second and return the quotient.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Dividend"},
                "b": {"type": "number", "description": "Divisor (must not be zero)"},
            },
            "required": ["a", "b"],
        },
        callable=_divide,
    ),
    Tool(
        name="raise_to_power",
        description="Raise a base number to an exponent and return the result.",
        parameters={
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "Base number"},
                "exponent": {"type": "number", "description": "Exponent"},
            },
            "required": ["base", "exponent"],
        },
        callable=_power,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 2. Unit conversion — length                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

LENGTH_TOOLS = [
    Tool(
        name="convert_km_to_miles",
        description="Convert a distance in kilometres to miles.",
        parameters={
            "type": "object",
            "properties": {"km": {"type": "number", "description": "Distance in kilometres"}},
            "required": ["km"],
        },
        callable=lambda km: round(km * 0.621371, 6),
    ),
    Tool(
        name="convert_miles_to_km",
        description="Convert a distance in miles to kilometres.",
        parameters={
            "type": "object",
            "properties": {"miles": {"type": "number", "description": "Distance in miles"}},
            "required": ["miles"],
        },
        callable=lambda miles: round(miles * 1.60934, 6),
    ),
    Tool(
        name="convert_meters_to_feet",
        description="Convert a length in metres to feet.",
        parameters={
            "type": "object",
            "properties": {"meters": {"type": "number", "description": "Length in metres"}},
            "required": ["meters"],
        },
        callable=lambda meters: round(meters * 3.28084, 6),
    ),
    Tool(
        name="convert_feet_to_meters",
        description="Convert a length in feet to metres.",
        parameters={
            "type": "object",
            "properties": {"feet": {"type": "number", "description": "Length in feet"}},
            "required": ["feet"],
        },
        callable=lambda feet: round(feet * 0.3048, 6),
    ),
    Tool(
        name="convert_inches_to_cm",
        description="Convert a length in inches to centimetres.",
        parameters={
            "type": "object",
            "properties": {"inches": {"type": "number", "description": "Length in inches"}},
            "required": ["inches"],
        },
        callable=lambda inches: round(inches * 2.54, 6),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 3. Unit conversion — weight / mass  (confusable with #2)                    #
# ═══════════════════════════════════════════════════════════════════════════ #

WEIGHT_TOOLS = [
    Tool(
        name="convert_kg_to_lbs",
        description="Convert a weight in kilograms to pounds.",
        parameters={
            "type": "object",
            "properties": {"kg": {"type": "number", "description": "Weight in kilograms"}},
            "required": ["kg"],
        },
        callable=lambda kg: round(kg * 2.20462, 6),
    ),
    Tool(
        name="convert_lbs_to_kg",
        description="Convert a weight in pounds to kilograms.",
        parameters={
            "type": "object",
            "properties": {"lbs": {"type": "number", "description": "Weight in pounds"}},
            "required": ["lbs"],
        },
        callable=lambda lbs: round(lbs * 0.453592, 6),
    ),
    Tool(
        name="convert_grams_to_ounces",
        description="Convert a mass in grams to ounces.",
        parameters={
            "type": "object",
            "properties": {"grams": {"type": "number", "description": "Mass in grams"}},
            "required": ["grams"],
        },
        callable=lambda grams: round(grams * 0.035274, 6),
    ),
    Tool(
        name="convert_ounces_to_grams",
        description="Convert a mass in ounces to grams.",
        parameters={
            "type": "object",
            "properties": {"ounces": {"type": "number", "description": "Mass in ounces"}},
            "required": ["ounces"],
        },
        callable=lambda ounces: round(ounces * 28.3495, 6),
    ),
    Tool(
        name="convert_tonnes_to_kg",
        description="Convert a mass in metric tonnes to kilograms.",
        parameters={
            "type": "object",
            "properties": {"tonnes": {"type": "number", "description": "Mass in metric tonnes"}},
            "required": ["tonnes"],
        },
        callable=lambda tonnes: round(tonnes * 1000, 6),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 4. Temperature conversion                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

TEMP_TOOLS = [
    Tool(
        name="celsius_to_fahrenheit",
        description="Convert a temperature from Celsius to Fahrenheit.",
        parameters={
            "type": "object",
            "properties": {"celsius": {"type": "number", "description": "Temperature in Celsius"}},
            "required": ["celsius"],
        },
        callable=lambda celsius: round(celsius * 9 / 5 + 32, 4),
    ),
    Tool(
        name="fahrenheit_to_celsius",
        description="Convert a temperature from Fahrenheit to Celsius.",
        parameters={
            "type": "object",
            "properties": {"fahrenheit": {"type": "number", "description": "Temperature in Fahrenheit"}},
            "required": ["fahrenheit"],
        },
        callable=lambda fahrenheit: round((fahrenheit - 32) * 5 / 9, 4),
    ),
    Tool(
        name="celsius_to_kelvin",
        description="Convert a temperature from Celsius to Kelvin.",
        parameters={
            "type": "object",
            "properties": {"celsius": {"type": "number", "description": "Temperature in Celsius"}},
            "required": ["celsius"],
        },
        callable=lambda celsius: round(celsius + 273.15, 4),
    ),
    Tool(
        name="kelvin_to_celsius",
        description="Convert a temperature from Kelvin to Celsius.",
        parameters={
            "type": "object",
            "properties": {"kelvin": {"type": "number", "description": "Temperature in Kelvin"}},
            "required": ["kelvin"],
        },
        callable=lambda kelvin: round(kelvin - 273.15, 4),
    ),
    Tool(
        name="fahrenheit_to_kelvin",
        description="Convert a temperature from Fahrenheit to Kelvin.",
        parameters={
            "type": "object",
            "properties": {"fahrenheit": {"type": "number", "description": "Temperature in Fahrenheit"}},
            "required": ["fahrenheit"],
        },
        callable=lambda fahrenheit: round((fahrenheit - 32) * 5 / 9 + 273.15, 4),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 5. Date & time                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def _days_between(date1: str, date2: str) -> int:
    d1 = datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.strptime(date2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def _add_days(date: str, days: int) -> str:
    d = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=days)
    return d.strftime("%Y-%m-%d")

def _day_of_week(date: str) -> str:
    return datetime.strptime(date, "%Y-%m-%d").strftime("%A")

def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def _week_number(date: str) -> int:
    return datetime.strptime(date, "%Y-%m-%d").isocalendar()[1]

DATE_TOOLS = [
    Tool(
        name="days_between_dates",
        description="Calculate the number of days between two calendar dates.",
        parameters={
            "type": "object",
            "properties": {
                "date1": {"type": "string", "description": "First date (YYYY-MM-DD)"},
                "date2": {"type": "string", "description": "Second date (YYYY-MM-DD)"},
            },
            "required": ["date1", "date2"],
        },
        callable=_days_between,
    ),
    Tool(
        name="add_days_to_date",
        description="Add a number of days to a date and return the resulting date.",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Starting date (YYYY-MM-DD)"},
                "days": {"type": "integer", "description": "Number of days to add (can be negative)"},
            },
            "required": ["date", "days"],
        },
        callable=_add_days,
    ),
    Tool(
        name="get_day_of_week",
        description="Return the day of the week (e.g. Monday) for a given date.",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string", "description": "Date (YYYY-MM-DD)"}},
            "required": ["date"],
        },
        callable=_day_of_week,
    ),
    Tool(
        name="is_leap_year",
        description="Check whether a given year is a leap year.",
        parameters={
            "type": "object",
            "properties": {"year": {"type": "integer", "description": "Year to check"}},
            "required": ["year"],
        },
        callable=_is_leap_year,
    ),
    Tool(
        name="get_iso_week_number",
        description="Return the ISO week number (1–53) for a given date.",
        parameters={
            "type": "object",
            "properties": {"date": {"type": "string", "description": "Date (YYYY-MM-DD)"}},
            "required": ["date"],
        },
        callable=_week_number,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 6. Currency conversion (mocked rates, 2024-01-01 snapshot)                  #
# ═══════════════════════════════════════════════════════════════════════════ #

_RATES_TO_USD = {
    "USD": 1.0, "EUR": 1.10, "GBP": 1.27, "JPY": 0.0067,
    "INR": 0.012, "CAD": 0.74, "AUD": 0.65, "CHF": 1.15,
    "CNY": 0.14, "MXN": 0.059,
}

def _convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    fc = from_currency.upper()
    tc = to_currency.upper()
    if fc not in _RATES_TO_USD:
        raise ValueError(f"Unknown currency: {fc}")
    if tc not in _RATES_TO_USD:
        raise ValueError(f"Unknown currency: {tc}")
    usd = amount * _RATES_TO_USD[fc]
    return round(usd / _RATES_TO_USD[tc], 4)

def _list_currencies() -> list:
    return sorted(_RATES_TO_USD.keys())

def _usd_to_eur(amount: float) -> float:
    return round(amount / _RATES_TO_USD["EUR"], 4)

def _eur_to_usd(amount: float) -> float:
    return round(amount * _RATES_TO_USD["EUR"], 4)

def _exchange_rate(from_currency: str, to_currency: str) -> float:
    fc = from_currency.upper()
    tc = to_currency.upper()
    if fc not in _RATES_TO_USD or tc not in _RATES_TO_USD:
        raise ValueError("Unknown currency")
    return round(_RATES_TO_USD[fc] / _RATES_TO_USD[tc], 6)

CURRENCY_TOOLS = [
    Tool(
        name="convert_currency",
        description="Convert an amount from one currency to another using fixed 2024 exchange rates.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code (e.g. USD)"},
                "to_currency": {"type": "string", "description": "Target currency code (e.g. EUR)"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
        callable=_convert_currency,
    ),
    Tool(
        name="list_supported_currencies",
        description="Return a list of all currency codes supported by the currency converter.",
        parameters={"type": "object", "properties": {}, "required": []},
        callable=_list_currencies,
    ),
    Tool(
        name="convert_usd_to_eur",
        description="Convert an amount from US Dollars to Euros.",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "number", "description": "Amount in USD"}},
            "required": ["amount"],
        },
        callable=_usd_to_eur,
    ),
    Tool(
        name="convert_eur_to_usd",
        description="Convert an amount from Euros to US Dollars.",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "number", "description": "Amount in EUR"}},
            "required": ["amount"],
        },
        callable=_eur_to_usd,
    ),
    Tool(
        name="get_exchange_rate",
        description="Get the exchange rate between two currencies (how many units of to_currency per 1 unit of from_currency).",
        parameters={
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "Source currency code"},
                "to_currency": {"type": "string", "description": "Target currency code"},
            },
            "required": ["from_currency", "to_currency"],
        },
        callable=_exchange_rate,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 7. String manipulation                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

STRING_TOOLS = [
    Tool(
        name="word_count",
        description="Count the number of words in a text string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Input text"}},
            "required": ["text"],
        },
        callable=lambda text: len(text.split()),
    ),
    Tool(
        name="character_count",
        description="Count the total number of characters in a text string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Input text"}},
            "required": ["text"],
        },
        callable=lambda text: len(text),
    ),
    Tool(
        name="reverse_string",
        description="Reverse the characters in a string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Input text to reverse"}},
            "required": ["text"],
        },
        callable=lambda text: text[::-1],
    ),
    Tool(
        name="to_uppercase",
        description="Convert all characters in a string to uppercase.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Input text"}},
            "required": ["text"],
        },
        callable=lambda text: text.upper(),
    ),
    Tool(
        name="count_vowels",
        description="Count the number of vowel characters (a, e, i, o, u) in a string.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Input text"}},
            "required": ["text"],
        },
        callable=lambda text: sum(1 for c in text.lower() if c in "aeiou"),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 8. Statistics                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def _mean(numbers: list) -> float:
    return statistics.mean(numbers)

def _median(numbers: list) -> float:
    return statistics.median(numbers)

def _stdev(numbers: list) -> float:
    if len(numbers) < 2:
        raise ValueError("Standard deviation requires at least 2 values")
    return round(statistics.stdev(numbers), 8)

def _list_max(numbers: list) -> float:
    return max(numbers)

def _list_min(numbers: list) -> float:
    return min(numbers)

STATS_TOOLS = [
    Tool(
        name="calculate_mean",
        description="Calculate the arithmetic mean (average) of a list of numbers.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers",
                }
            },
            "required": ["numbers"],
        },
        callable=_mean,
    ),
    Tool(
        name="calculate_median",
        description="Calculate the median (middle value) of a list of numbers.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers",
                }
            },
            "required": ["numbers"],
        },
        callable=_median,
    ),
    Tool(
        name="calculate_standard_deviation",
        description="Calculate the sample standard deviation of a list of numbers.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers (at least 2)",
                }
            },
            "required": ["numbers"],
        },
        callable=_stdev,
    ),
    Tool(
        name="find_maximum",
        description="Find the largest value in a list of numbers.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers",
                }
            },
            "required": ["numbers"],
        },
        callable=_list_max,
    ),
    Tool(
        name="find_minimum",
        description="Find the smallest value in a list of numbers.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers",
                }
            },
            "required": ["numbers"],
        },
        callable=_list_min,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 9. Geocoding & geography (mocked)                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

_CITY_DATA = {
    "Tokyo":      {"lat": 35.6762, "lon": 139.6503, "country": "Japan",          "population": 13960000},
    "Delhi":      {"lat": 28.7041, "lon": 77.1025,  "country": "India",          "population": 32226000},
    "Shanghai":   {"lat": 31.2304, "lon": 121.4737, "country": "China",          "population": 24870000},
    "São Paulo":  {"lat": -23.5505,"lon": -46.6333, "country": "Brazil",         "population": 22430000},
    "Mumbai":     {"lat": 19.0760, "lon": 72.8777,  "country": "India",          "population": 20667000},
    "New York":   {"lat": 40.7128, "lon": -74.0060, "country": "United States",  "population": 8336817},
    "London":     {"lat": 51.5074, "lon": -0.1278,  "country": "United Kingdom", "population": 9648110},
    "Paris":      {"lat": 48.8566, "lon": 2.3522,   "country": "France",         "population": 2141000},
    "Berlin":     {"lat": 52.5200, "lon": 13.4050,  "country": "Germany",        "population": 3769000},
    "Sydney":     {"lat": -33.8688,"lon": 151.2093, "country": "Australia",      "population": 5312000},
    "Toronto":    {"lat": 43.6532, "lon": -79.3832, "country": "Canada",         "population": 2930000},
    "Cairo":      {"lat": 30.0444, "lon": 31.2357,  "country": "Egypt",          "population": 21323000},
}

_COUNTRY_DATA = {
    "Japan":         {"area_km2": 377975,  "population": 125700000, "capital": "Tokyo"},
    "India":         {"area_km2": 3287263, "population": 1428600000,"capital": "New Delhi"},
    "China":         {"area_km2": 9596960, "population": 1412600000,"capital": "Beijing"},
    "Brazil":        {"area_km2": 8515767, "population": 215313498, "capital": "Brasília"},
    "United States": {"area_km2": 9833517, "population": 331000000, "capital": "Washington D.C."},
    "United Kingdom":{"area_km2": 242495,  "population": 67220000,  "capital": "London"},
    "France":        {"area_km2": 551695,  "population": 68000000,  "capital": "Paris"},
    "Germany":       {"area_km2": 357114,  "population": 83200000,  "capital": "Berlin"},
    "Australia":     {"area_km2": 7692024, "population": 25700000,  "capital": "Canberra"},
    "Canada":        {"area_km2": 9984670, "population": 38250000,  "capital": "Ottawa"},
    "Egypt":         {"area_km2": 1002450, "population": 104258000, "capital": "Cairo"},
}

def _geocode_city(city: str) -> dict:
    data = _CITY_DATA.get(city)
    if data is None:
        raise ValueError(f"City not found: {city!r}. Available: {list(_CITY_DATA)}")
    return {"city": city, "lat": data["lat"], "lon": data["lon"], "country": data["country"]}

def _get_city_population(city: str) -> int:
    data = _CITY_DATA.get(city)
    if data is None:
        raise ValueError(f"City not found: {city!r}")
    return data["population"]

def _get_country_population(country: str) -> int:
    data = _COUNTRY_DATA.get(country)
    if data is None:
        raise ValueError(f"Country not found: {country!r}")
    return data["population"]

def _get_country_area(country: str) -> float:
    data = _COUNTRY_DATA.get(country)
    if data is None:
        raise ValueError(f"Country not found: {country!r}")
    return data["area_km2"]

def _get_country_capital(country: str) -> str:
    data = _COUNTRY_DATA.get(country)
    if data is None:
        raise ValueError(f"Country not found: {country!r}")
    return data["capital"]

GEO_TOOLS = [
    Tool(
        name="geocode_city",
        description="Return the latitude and longitude coordinates for a major world city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name (e.g. Tokyo)"}},
            "required": ["city"],
        },
        callable=_geocode_city,
    ),
    Tool(
        name="get_city_population",
        description="Return the population of a major world city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
        callable=_get_city_population,
    ),
    Tool(
        name="get_country_population",
        description="Return the total population of a country.",
        parameters={
            "type": "object",
            "properties": {"country": {"type": "string", "description": "Country name"}},
            "required": ["country"],
        },
        callable=_get_country_population,
    ),
    Tool(
        name="get_country_area",
        description="Return the land area of a country in square kilometres.",
        parameters={
            "type": "object",
            "properties": {"country": {"type": "string", "description": "Country name"}},
            "required": ["country"],
        },
        callable=_get_country_area,
    ),
    Tool(
        name="get_country_capital",
        description="Return the capital city of a country.",
        parameters={
            "type": "object",
            "properties": {"country": {"type": "string", "description": "Country name"}},
            "required": ["country"],
        },
        callable=_get_country_capital,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 10. General knowledge / trivia (mocked)                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

_ELEMENT_DATA = {
    "hydrogen":  {"symbol": "H",  "atomic_number": 1,   "atomic_weight": 1.008},
    "helium":    {"symbol": "He", "atomic_number": 2,   "atomic_weight": 4.003},
    "carbon":    {"symbol": "C",  "atomic_number": 6,   "atomic_weight": 12.011},
    "nitrogen":  {"symbol": "N",  "atomic_number": 7,   "atomic_weight": 14.007},
    "oxygen":    {"symbol": "O",  "atomic_number": 8,   "atomic_weight": 15.999},
    "sodium":    {"symbol": "Na", "atomic_number": 11,  "atomic_weight": 22.990},
    "iron":      {"symbol": "Fe", "atomic_number": 26,  "atomic_weight": 55.845},
    "gold":      {"symbol": "Au", "atomic_number": 79,  "atomic_weight": 196.967},
    "silver":    {"symbol": "Ag", "atomic_number": 47,  "atomic_weight": 107.868},
    "uranium":   {"symbol": "U",  "atomic_number": 92,  "atomic_weight": 238.029},
}

_PLANET_DATA = {
    "Mercury": {"order": 1, "diameter_km": 4879,   "moons": 0,   "distance_from_sun_AU": 0.39},
    "Venus":   {"order": 2, "diameter_km": 12104,  "moons": 0,   "distance_from_sun_AU": 0.72},
    "Earth":   {"order": 3, "diameter_km": 12756,  "moons": 1,   "distance_from_sun_AU": 1.00},
    "Mars":    {"order": 4, "diameter_km": 6792,   "moons": 2,   "distance_from_sun_AU": 1.52},
    "Jupiter": {"order": 5, "diameter_km": 142984, "moons": 95,  "distance_from_sun_AU": 5.20},
    "Saturn":  {"order": 6, "diameter_km": 120536, "moons": 146, "distance_from_sun_AU": 9.58},
    "Uranus":  {"order": 7, "diameter_km": 51118,  "moons": 28,  "distance_from_sun_AU": 19.2},
    "Neptune": {"order": 8, "diameter_km": 49528,  "moons": 16,  "distance_from_sun_AU": 30.0},
}

def _element_info(element_name: str) -> dict:
    data = _ELEMENT_DATA.get(element_name.lower())
    if data is None:
        raise ValueError(f"Unknown element: {element_name!r}")
    return {"name": element_name, **data}

def _planet_moons(planet: str) -> int:
    data = _PLANET_DATA.get(planet.capitalize())
    if data is None:
        raise ValueError(f"Unknown planet: {planet!r}")
    return data["moons"]

def _planet_diameter(planet: str) -> int:
    data = _PLANET_DATA.get(planet.capitalize())
    if data is None:
        raise ValueError(f"Unknown planet: {planet!r}")
    return data["diameter_km"]

def _speed_of_light_in_medium(medium: str) -> float:
    speeds = {
        "vacuum": 299792458.0,
        "air": 299702458.0,
        "water": 224900000.0,
        "glass": 199861638.0,
        "diamond": 123950000.0,
    }
    s = speeds.get(medium.lower())
    if s is None:
        raise ValueError(f"Unknown medium: {medium!r}. Known: {list(speeds)}")
    return s

def _boiling_point(substance: str) -> dict:
    data = {
        "water":    {"celsius": 100.0,   "fahrenheit": 212.0},
        "ethanol":  {"celsius": 78.37,   "fahrenheit": 173.1},
        "nitrogen": {"celsius": -195.79, "fahrenheit": -320.4},
        "oxygen":   {"celsius": -182.96, "fahrenheit": -297.3},
        "mercury":  {"celsius": 356.73,  "fahrenheit": 674.1},
    }
    d = data.get(substance.lower())
    if d is None:
        raise ValueError(f"Unknown substance: {substance!r}")
    return {"substance": substance, **d}

TRIVIA_TOOLS = [
    Tool(
        name="get_element_info",
        description="Return the chemical symbol, atomic number, and atomic weight of a chemical element by name.",
        parameters={
            "type": "object",
            "properties": {
                "element_name": {"type": "string", "description": "Element name in English (e.g. gold)"}
            },
            "required": ["element_name"],
        },
        callable=_element_info,
    ),
    Tool(
        name="get_planet_moon_count",
        description="Return the number of known moons of a planet in our solar system.",
        parameters={
            "type": "object",
            "properties": {"planet": {"type": "string", "description": "Planet name (e.g. Jupiter)"}},
            "required": ["planet"],
        },
        callable=_planet_moons,
    ),
    Tool(
        name="get_planet_diameter",
        description="Return the equatorial diameter in kilometres of a planet in our solar system.",
        parameters={
            "type": "object",
            "properties": {"planet": {"type": "string", "description": "Planet name (e.g. Saturn)"}},
            "required": ["planet"],
        },
        callable=_planet_diameter,
    ),
    Tool(
        name="get_speed_of_light_in_medium",
        description="Return the approximate speed of light in metres per second for a given medium (vacuum, air, water, glass, diamond).",
        parameters={
            "type": "object",
            "properties": {
                "medium": {"type": "string", "description": "Medium name (vacuum, air, water, glass, diamond)"}
            },
            "required": ["medium"],
        },
        callable=_speed_of_light_in_medium,
    ),
    Tool(
        name="get_boiling_point",
        description="Return the boiling point in Celsius and Fahrenheit for a common substance (water, ethanol, nitrogen, oxygen, mercury).",
        parameters={
            "type": "object",
            "properties": {
                "substance": {"type": "string", "description": "Substance name (e.g. water)"}
            },
            "required": ["substance"],
        },
        callable=_boiling_point,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# Registry builder                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

ALL_TOOLS: List[Tool] = (
    MATH_TOOLS
    + LENGTH_TOOLS
    + WEIGHT_TOOLS
    + TEMP_TOOLS
    + DATE_TOOLS
    + CURRENCY_TOOLS
    + STRING_TOOLS
    + STATS_TOOLS
    + GEO_TOOLS
    + TRIVIA_TOOLS
)

assert len(ALL_TOOLS) == 50, f"Expected 50 tools, got {len(ALL_TOOLS)}"


def build_registry(n_tools: int = 50) -> ToolRegistry:
    """Build a ToolRegistry from the first *n_tools* tools.

    The 50 tools are ordered such that they span diverse domains even at
    smaller registry sizes:
      - 10 tools → math(5) + length(5)
      - 25 tools → above + weight(5) + temp(5) + date(5)
      - 50 tools → all 10 domains

    For the study, pass n_tools ∈ {10, 25, 50}.
    """
    if not (1 <= n_tools <= 50):
        raise ValueError(f"n_tools must be between 1 and 50, got {n_tools}")
    registry = ToolRegistry()
    registry.register_many(ALL_TOOLS[:n_tools])
    return registry


if __name__ == "__main__":
    # Quick sanity check
    r = build_registry(50)
    print(f"Registry size: {len(r)}")
    for t in r.all_tools():
        print(f"  {t.name}: {t.description[:60]}")
