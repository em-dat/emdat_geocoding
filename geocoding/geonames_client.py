"""Rate-limited client for the GeoNames search web service (searchJSON)."""
import datetime
import time

import numpy as np
import pandas as pd
import pycountry
import requests


class GeoNamesClient:
    def __init__(self, username, max_requests_per_hour=900, max_requests_per_day=20000):
        self.username = username
        self.max_requests_per_hour = max_requests_per_hour
        self.max_requests_per_day = max_requests_per_day
        
        self.hour_request_count = 0
        self.day_request_count = 0
        
        self.hour_start_time = datetime.datetime.now()
        self.day_start_time = datetime.datetime.now()
        
        self.delay_between_requests = 3600 / self.max_requests_per_hour

    def _wait_if_needed(self):
        now = datetime.datetime.now()
        
        # Check day limit
        if (now - self.day_start_time).total_seconds() >= 86400:
            self.day_request_count = 0
            self.day_start_time = now
            
        if self.day_request_count >= self.max_requests_per_day:
            sleep_time = 86400 - (now - self.day_start_time).total_seconds()
            if sleep_time > 0:
                print(f"Daily rate limit reached. Waiting for {sleep_time/3600:.2f} hours...")
                time.sleep(sleep_time)
            self.day_request_count = 0
            self.day_start_time = datetime.datetime.now()

        # Check hour limit
        now = datetime.datetime.now()
        if (now - self.hour_start_time).total_seconds() >= 3600:
            self.hour_request_count = 0
            self.hour_start_time = now
            
        if self.hour_request_count >= self.max_requests_per_hour:
            sleep_time = 3600 - (now - self.hour_start_time).total_seconds()
            if sleep_time > 0:
                print(f"Hourly rate limit reached. Waiting for {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
            self.hour_request_count = 0
            self.hour_start_time = datetime.datetime.now()

    def get_iso2(self, iso3):
        if not iso3 or not isinstance(iso3, str):
            return None
        try:
            country = pycountry.countries.get(alpha_3=iso3.upper())
            if country:
                return country.alpha_2
        except Exception:
            pass
        return None

    def search(self, query, iso3=None, fuzzy=0.7):
        self._wait_if_needed()
        
        params = {
            "q": query,
            "maxRows": 1,
            "username": self.username,
            "fuzzy": fuzzy
        }
        
        if iso3:
            iso2 = self.get_iso2(iso3)
            if iso2:
                params["country"] = iso2
            else:
                # If ISO3 is SDN, try with SSD (South Sudan) as per original logic
                if iso3.upper() == "SDN":
                    params["country"] = "SS"
        
        try:
            response = requests.get("http://api.geonames.org/searchJSON", params=params, timeout=10)
            self.hour_request_count += 1
            self.day_request_count += 1
            
            # Minimum delay between requests to be safe
            time.sleep(max(0.1, self.delay_between_requests))
            
            response.raise_for_status()
            data = response.json()
            
            if "geonames" in data and data["geonames"]:
                result = data["geonames"][0]
                return {
                    "name": result.get("name"),
                    "lat": float(result.get("lat")),
                    "lng": float(result.get("lng")),
                    "adminName1": result.get("adminName1"),
                    "countryCode": result.get("countryCode"),
                    "geonameId": result.get("geonameId")
                }
            return None
        except Exception as e:
            print(f"Error calling GeoNames for '{query}': {e}")
            return None

def geocode_event_locations(df, username, max_requests_per_hour=900, max_requests_per_day=20000):
    """
    Geocode a DataFrame of locations using GeoNames.
    Expected columns: 'DisNo.', 'ISO', 'Location'
    """
    client = GeoNamesClient(username=username,
                            max_requests_per_hour=max_requests_per_hour,
                            max_requests_per_day=max_requests_per_day)
    results = []
    
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f"Geocoding progress: {i}/{total}")
            
        dis_no = row["DisNo."]
        iso = row["ISO"]
        location_name = row["Location"]
        
        geo_result = client.search(location_name, iso3=iso)
        
        if geo_result:
            results.append({
                "DisNo.": dis_no,
                "input_location": location_name,
                "geoname_name": geo_result["name"],
                "lat": geo_result["lat"],
                "lng": geo_result["lng"],
                "admin1_name": geo_result["adminName1"],
                "iso2": geo_result["countryCode"],
                "geoname_id": geo_result["geonameId"]
            })
        else:
            # Add entry with NAs if geocoding failed
            results.append({
                "DisNo.": dis_no,
                "input_location": location_name,
                "geoname_name": np.nan,
                "lat": np.nan,
                "lng": np.nan,
                "admin1_name": np.nan,
                "iso2": np.nan,
                "geoname_id": np.nan
            })
            
    return pd.DataFrame(results)
