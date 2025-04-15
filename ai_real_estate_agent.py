import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import requests
from openai import OpenAI
import json
from supabase import create_client, Client
import os

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Real Estate Agent", layout="wide")

# --- Supabase Global Variables ---
# Define globals, but initialize client inside main()
supabase: Optional[Client] = None
supabase_connected = False
KEYS_TABLE = "api_keys"
KEYS_ID = 1

# --- Función para inicializar Supabase ---
def initialize_supabase():
    try:
        supabase_url = st.secrets["supabase_url"]
        supabase_key = st.secrets["supabase_key"]
        return create_client(supabase_url, supabase_key)
    except KeyError:
        st.error("Supabase URL/Key not found in Streamlit secrets. Please configure .streamlit/secrets.toml")
        return None
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        return None

# --- Modelos Pydantic ---
class PropertyData(BaseModel):
    building_name: str = Field(..., description="Nombre del edificio o complejo")
    property_type: str = Field(..., description="Tipo de propiedad")
    location_address: str = Field(..., description="Dirección completa")
    price: float = Field(..., description="Precio en USD")
    description: str = Field(..., description="Descripción detallada")
    square_feet: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None

class LocationsResponse(BaseModel):
    location: str
    average_price: float
    price_trend: List[float]
    market_demand: str

# --- Funciones Supabase ---
def save_keys_to_supabase(rapidapi_key: str, openai_key: str) -> bool:
    supabase_client = initialize_supabase()
    if not supabase_client:
        return False
    try:
        data = supabase_client.table(KEYS_TABLE).upsert({ 
            "id": KEYS_ID,
            "rapidapi_key": rapidapi_key,
            "openai_key": openai_key
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error saving keys to Supabase: {e}")
        return False

def get_keys_from_supabase() -> Optional[dict]:
    supabase_client = initialize_supabase()
    if not supabase_client:
        return None
    try:
        response = supabase_client.table(KEYS_TABLE).select("rapidapi_key, openai_key").eq("id", KEYS_ID).execute()
        if response.data and len(response.data) > 0:
            keys = response.data[0]
            if keys.get("rapidapi_key") and keys.get("openai_key"):
                return keys
            else:
                st.warning("DB row found, but missing expected rapidapi_key or openai_key.")
                return None
        else:
            st.info("No API keys found in Supabase for the specified ID.")
            return None
    except Exception as e:
        st.error(f"Exception occurred retrieving keys from Supabase: {e}")
        return None

# --- Funciones Auxiliares ---
def generate_summary(properties: List[PropertyData], client: OpenAI) -> str:
    if not properties:
        return "No hay propiedades para resumir."

    prompt_details = "\n".join([
        f"- {p.building_name or 'Propiedad'} ({p.property_type or 'N/A'}) en {p.location_address or 'N/A'} por ${p.price:,.2f} ({p.bedrooms or 'N/A'} hab, {p.bathrooms or 'N/A'} baños)" 
        for p in properties
    ])

    system_prompt = "Eres un asistente inmobiliario. Resume concisamente las siguientes propiedades listadas, destacando el rango de precios, tipos comunes y ubicaciones principales."
    user_prompt = f"Aquí tienes una lista de propiedades:\n{prompt_details}\n\nPor favor, proporciona un resumen breve."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"Error al generar resumen con OpenAI: {e}")
        return "No se pudo generar el resumen."

# --- Nueva Función de Búsqueda con RapidAPI ---
def search_properties_rapidapi(latitude: float, longitude: float, radius: int, rapidapi_key: str) -> List[PropertyData]:
    """
    Busca propiedades usando el endpoint /search/forrent/coordinates de Realtor.com via RapidAPI y mapea la respuesta al modelo PropertyData.
    """
    api_host = "realtor16.p.rapidapi.com"
    api_endpoint = "/search/forrent/coordinates"
    api_url = f"https://{api_host}{api_endpoint}"

    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": api_host
    }

    querystring = {
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius
    }

    properties_data = []
    try:
        response = requests.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        api_response_json = response.json()

        results_list = api_response_json.get("properties", [])
        if not results_list:
            st.warning("La API no devolvió resultados en la estructura esperada (no se encontró 'properties') o la lista está vacía.")
            st.subheader("Respuesta cruda de la API:")
            st.json(api_response_json)
            return []

        for prop_api in results_list:
            try:
                address = prop_api.get("location", {}).get("address", {})
                description = prop_api.get("description", {})
                price = prop_api.get("list_price") or 0.0
                def safe_float(val, default=None):
                    try:
                        if isinstance(val, str):
                            val = val.replace('+', '').strip()
                        return float(val)
                    except (ValueError, TypeError):
                        return default

                property_data = PropertyData(
                    building_name=address.get("line") or "N/A",
                    property_type=description.get("type") or "N/A",
                    location_address=f"{address.get('line') or ''}, {address.get('city') or ''}, {address.get('state_code') or ''} {address.get('postal_code') or ''}",
                    price=price,
                    description=f"{description.get('beds', 'N/A')} hab, {description.get('baths_consolidated', 'N/A')} baños, {description.get('sqft', 'N/A')} sqft",
                    square_feet=description.get("sqft"),
                    bedrooms=description.get("beds"),
                    bathrooms=safe_float(description.get("baths_consolidated"))
                )
                properties_data.append(property_data)
            except Exception as map_e:
                st.error(f"Error inesperado al mapear propiedad: {map_e} - Data: {prop_api}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de red o HTTP al llamar a RapidAPI: {e}")
    except json.JSONDecodeError:
        st.error("Error: La respuesta de RapidAPI no es un JSON válido.")
        st.text(response.text)
    except Exception as e:
        st.error(f"Error inesperado durante la búsqueda con RapidAPI: {e}")

    return properties_data

# --- Interfaz Streamlit ---
def main():
    global supabase, supabase_connected
    try:
        supabase_url = st.secrets["supabase_url"]
        supabase_key = st.secrets["supabase_key"]
        supabase = create_client(supabase_url, supabase_key)
        supabase_connected = True
    except KeyError:
        st.error("Supabase URL/Key not found in Streamlit secrets. Please configure .streamlit/secrets.toml")
        supabase_connected = False
        st.stop()
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        supabase_connected = False
        st.stop()

    st.title("AI Real Estate Agent")

    api_keys = None
    if supabase_connected:
        api_keys = get_keys_from_supabase()

    if api_keys:
        try:
            openai_client = OpenAI(api_key=api_keys["openai_key"])
            st.success("Cliente OpenAI inicializado correctamente.")
        except Exception as e:
            st.error(f"Fallo al inicializar el cliente OpenAI: {e}")
            return

        st.sidebar.header("API Keys Configuration")
        st.sidebar.header("Property Filters")
        max_price = st.sidebar.number_input("Maximum Price ($)", min_value=0, value=0, step=10000)
        min_bedrooms = st.sidebar.number_input("Minimum Bedrooms", min_value=0, value=0, step=1)
        min_bathrooms = st.sidebar.number_input("Minimum Bathrooms", min_value=0, value=0, step=1)
        property_type_options = ["Any", "Apartment", "House", "Condo", "Townhouse", "Land", "Other"]
        property_type_filter = st.sidebar.selectbox("Property Type", options=property_type_options)

        st.header("Search Properties")

        # --- MAP PICKER ---
        from streamlit_elements import elements, mui, html, sync, event, lazy, dashboard
        import requests
        import json
        if "selected_coords" not in st.session_state:
            st.session_state.selected_coords = {"lat": 29.27052, "lon": -95.74991}
        if "selected_address" not in st.session_state:
            st.session_state.selected_address = ""

        # --- City/address autocomplete search ---
        city_query = st.text_input("City or Address Search (autocomplete)", "", key="city_search_input")
        city_suggestions = []
        if city_query and len(city_query) > 2:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": city_query, "format": "json", "addressdetails": 1, "limit": 5, "accept-language": "en"},
                    headers={"User-Agent": "streamlit-realestate-app"}
                )
                if resp.status_code == 200:
                    city_suggestions = resp.json()
            except Exception:
                pass
        city_names = [f"{c.get('display_name', '')}" for c in city_suggestions]
        selected_city = st.selectbox("Choose a location from suggestions:", city_names, index=0 if city_names else None, key="city_search_select") if city_names else None
        # When a city is selected, update coordinates
        if selected_city:
            idx = city_names.index(selected_city)
            lat = float(city_suggestions[idx]["lat"])
            lon = float(city_suggestions[idx]["lon"])
            st.session_state.selected_coords["lat"] = lat
            st.session_state.selected_coords["lon"] = lon
            # Reverse geocode for address
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={"lat": lat, "lon": lon, "format": "json", "accept-language": "en"},
                    headers={"User-Agent": "streamlit-realestate-app"}
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.selected_address = result.get("display_name", "")
            except Exception:
                st.session_state.selected_address = ""

    
        # --- Display coordinates and address ---
        st.write(f"**Address:** {st.session_state.selected_address}")
        st.header("Search properties by coordinates")
        latitude = st.number_input("Latitude", value=st.session_state.selected_coords["lat"], format="%.5f", key="latitude_input")
        longitude = st.number_input("Longitude", value=st.session_state.selected_coords["lon"], format="%.5f", key="longitude_input")
        # Sync map and inputs
        if latitude != st.session_state.selected_coords["lat"] or longitude != st.session_state.selected_coords["lon"]:
            st.session_state.selected_coords["lat"] = latitude
            st.session_state.selected_coords["lon"] = longitude
        radius_min, radius_max = st.slider("Search radius (km)", min_value=1, max_value=100, value=(10, 30), step=1)
        if st.button("Search Properties", key="search_properties_coords"):
            with st.spinner(f"Searching for properties near lat={latitude}, lon={longitude}, radius={radius_min}-{radius_max}km..."):
                rapidapi_key = api_keys["rapidapi_key"]
                property_results = search_properties_rapidapi(latitude, longitude, (radius_min, radius_max), rapidapi_key)
            if property_results:
                st.success(f"{len(property_results)} results found for lat={latitude}, lon={longitude}, radius={radius_min}-{radius_max}km via RapidAPI.")

                # --- ANÁLISIS IA EN LA PARTE SUPERIOR ---
                openai_key = api_keys.get("openai_key")
                if openai_key:
                    try:
                        # Preparar datos regionales simulados (puedes reemplazar con datos reales si los tienes)
                        regional_data = {
                            'price_per_sqft': 180,
                            'tax_rate': 1.8
                        }
                        # Formatear propiedades para el prompt
                        properties_str = "\n".join([
                            f"- {p.building_name} ({p.property_type}) en {p.location_address} por ${p.price:,.0f} ({p.bedrooms or 'N/A'} hab, {p.bathrooms or 'N/A'} baños, {p.square_feet or 'N/A'} sqft)"
                            for p in property_results[:10]
                        ])
                        prompt = f'''
As a U.S. real estate expert, analyze these properties:

Properties:
{properties_str}

**Analysis Requirements:**
1. Market Comparison:
   - Price vs local median (${{regional_data.get('price_per_sqft', 0)}}/sqft)
   - Days on market analysis
2. Property Evaluation:
   - Price/square foot
   - Lot size value
3. Financial Considerations:
   - Tax estimates ({{regional_data.get('tax_rate', 0)}}% rate)
   - HOA impact
4. Recommendations:
   - Top 3 value picks
   - Investment potential
   - Red flags

**Response Format:**

🏡 PROPERTY ANALYSIS
• Price/SqFt Comparison
• Key Value Indicators

📈 MARKET INSIGHTS
• Local Trends
• Inventory Analysis

💰 FINANCIAL BREAKDOWN
• Tax Estimates
• Maintenance Costs

🏆 TOP PICKS
• Best Value
• Best Investment

🔍 RED FLAGS
• Overpriced
• Structural Concerns
'''
                        openai_client = OpenAI(api_key=openai_key)
                        with st.spinner("Analizando propiedades con IA..."):
                            response = openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "system", "content": "You are a U.S. real estate expert."},
                                          {"role": "user", "content": prompt}],
                                temperature=0.5,
                                max_tokens=500
                            )
                        analysis = response.choices[0].message.content.strip()
                        st.markdown("### 🏡 AI Property Analysis")
                        with st.expander("View detailed AI analysis", expanded=True):
                            st.markdown(analysis)
                        # --- Investment Advantages Summary ---
                        resumen_prompt = f"""
As a professional real estate advisor, write a 3-5 sentence summary of the investment advantages of these properties, specifically addressed to Realtors and real estate sellers. Highlight useful arguments for client acquisition, such as appreciation potential, area attractiveness, rental demand, expected returns, and competitive advantages versus the market. Use a professional, clear, and consultative sales tone."""
                        try:
                            resumen_response = openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "system", "content": "You are a senior real estate advisor specialized in investments and B2B sales."},
                                          {"role": "user", "content": resumen_prompt + "\n\nAI ANALYSIS:\n" + analysis}],
                                temperature=0.4,
                                max_tokens=200
                            )
                            resumen_ventajas = resumen_response.choices[0].message.content.strip()
                            st.markdown("""
#### 💡 Professional summary for Realtors and sellers
""" + resumen_ventajas)
                        except Exception as e:
                            st.info(f"Could not generate investment advantages summary: {e}")
                        # --- Key Data Visualization ---
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        # Property table
                        df = pd.DataFrame([
                            {
                                "Address": p.location_address,
                                "Price": p.price,
                                "Bedrooms": p.bedrooms,
                                "Bathrooms": p.bathrooms,
                                "Area (sqft)": p.square_feet,
                                "Type": p.property_type
                            }
                            for p in property_results[:10]
                        ])
                        st.markdown("#### Featured comparative property table")
                        st.dataframe(df, use_container_width=True)
                        # Price chart
                        fig, ax = plt.subplots(figsize=(6,3))
                        df_sorted = df.sort_values("Price", ascending=False)
                        ax.barh(df_sorted["Address"], df_sorted["Price"], color="#388e3c")
                        ax.set_xlabel("Price ($)")
                        ax.set_title("Property price comparison")
                        st.pyplot(fig)
                        # Price per sqft chart
                        if df["Area (sqft)"].notnull().all() and (df["Area (sqft)"] > 0).all():
                            df["Price/sqft"] = df["Price"] / df["Area (sqft)"]
                            fig2, ax2 = plt.subplots(figsize=(6,3))
                            ax2.bar(df["Address"], df["Price/sqft"], color="#1976d2")
                            ax2.set_ylabel("Price per sqft ($)")
                            ax2.set_title("Price per square foot")
                            plt.xticks(rotation=45, ha="right")
                            st.pyplot(fig2)
                        st.info("The AI analysis is based on current data and may vary according to the market and available information. Use the charts and table to quickly compare the most attractive options.")
                    except Exception as e:
                        st.warning(f"No se pudo generar el análisis IA: {e}")

                # Show properties as visual cards
                cols = st.columns(2)
                for idx, prop in enumerate(property_results):
                    with cols[idx % 2]:
                        st.markdown("---")
                        # Show main photo if available
                        photo_url = None
                        if hasattr(prop, 'primary_photo') and prop.primary_photo and isinstance(prop.primary_photo, dict) and 'href' in prop.primary_photo:
                            photo_url = prop.primary_photo['href']
                        elif hasattr(prop, 'photos') and prop.photos and isinstance(prop.photos, list) and 'href' in prop.photos[0]:
                            photo_url = prop.photos[0]['href']
                        if photo_url:
                            st.image(photo_url, use_column_width=True)
                        st.markdown(f"**{prop.building_name}**")
                        st.markdown(f"{prop.location_address}")
                        st.markdown(f"<span style='color:#388e3c;font-size:1.2em'><b>${prop.price:,.0f}</b></span>", unsafe_allow_html=True)
                        st.markdown(f"Type: {prop.property_type}")
                        st.markdown(f"{prop.description}")
                        st.markdown(f"🛏️ {prop.bedrooms or '-'}  |  🛁 {prop.bathrooms or '-'}  |  📐 {prop.square_feet or '-'} sqft")
                        # Listing link
                        if hasattr(prop, 'permalink') and prop.permalink:
                            url = prop.permalink
                            if not url.startswith('http'):
                                url = f"https://www.realtor.com/realestateandhomes-detail/{url}"
                            st.markdown(f"[View listing on Realtor.com]({url})", unsafe_allow_html=True)
                        # Contact button (phone if available)
                        contact_phone = None
                        if hasattr(prop, 'advertisers') and prop.advertisers:
                            for adv in prop.advertisers:
                                if 'office' in adv and adv['office'] and 'phones' in adv['office'] and adv['office']['phones']:
                                    contact_phone = adv['office']['phones'][0]['number']
                                    break
                        if contact_phone:
                            st.markdown(f"<a href='tel:{contact_phone}'><button style='background:#1976d2;color:white;padding:6px 12px;border:none;border-radius:4px;cursor:pointer;'>Call {contact_phone}</button></a>", unsafe_allow_html=True)
            else:
                st.warning(f"No se encontraron propiedades en la búsqueda inicial para lat={latitude}, lon={longitude}, radio={radius}km.")

    else:
        st.warning("API Keys not found, incomplete, or Supabase connection failed.")
        st.subheader("Enter and Save API Keys")

        if not supabase_connected:
            st.error("Supabase is not connected. Cannot save or retrieve keys. Please check your credentials in .streamlit/secrets.toml and ensure the service is running.")
        else:
            with st.form("api_key_form"):
                rapidapi_key_input = st.text_input("RapidAPI Key", type="password")
                openai_key_input = st.text_input("OpenAI API Key", type="password")
                submitted = st.form_submit_button("Save Keys to Supabase")

                if submitted:
                    if rapidapi_key_input and openai_key_input:
                        if save_keys_to_supabase(rapidapi_key_input, openai_key_input):
                            st.sidebar.success("Claves guardadas exitosamente en Supabase!")
                            st.experimental_rerun()
                        else:
                            st.sidebar.error("Error al guardar claves en Supabase.")
                    else:
                        st.sidebar.warning("Por favor, introduce ambas claves API.")

if __name__ == "__main__":
    main()