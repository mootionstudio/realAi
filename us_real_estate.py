from typing import Dict, List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import streamlit as st
from functools import lru_cache

class PropertyData(BaseModel):
    """Schema for U.S. property data extraction"""
    address: str = Field(..., description="Full property address", alias="Address")
    property_type: str = Field(..., description="Type of property (Single-Family, Condo, etc)", alias="PropertyType")
    price: float = Field(..., description="Listing price in USD", alias="Price")
    square_footage: int = Field(..., description="Living area in square feet", alias="SqFt")
    bedrooms: int = Field(None, description="Number of bedrooms")
    bathrooms: float = Field(None, description="Number of bathrooms")
    lot_size: str = Field(None, description="Lot size in acres or sqft")
    year_built: int = Field(None, description="Year built")
    hoa_fees: str = Field(None, description="Monthly HOA fees")
    property_taxes: str = Field(None, description="Annual property taxes")
    mls_number: str = Field(None, description="MLS listing number")

class PropertiesResponse(BaseModel):
    """Schema for multiple properties response"""
    properties: List[PropertyData] = Field(description="List of property details")

class LocationTrends(BaseModel):
    """Schema for location price trends"""
    neighborhood: str
    median_price: float
    price_per_sqft: float
    yoy_change: float
    days_on_market: int
    school_rating: float

class LocationsResponse(BaseModel):
    """Schema for multiple locations response"""
    trends: List[LocationTrends] = Field(description="Neighborhood market trends")

class FirecrawlResponse(BaseModel):
    """Schema for Firecrawl API response"""
    success: bool
    data: Dict
    status: str
    expiresAt: str

REGIONAL_SETTINGS = {
    'CA': {'tax_rate': 0.012, 'price_per_sqft': 650},
    'TX': {'tax_rate': 0.021, 'price_per_sqft': 200},
    'FL': {'tax_rate': 0.019, 'price_per_sqft': 300},
    'NY': {'tax_rate': 0.015, 'price_per_sqft': 800},
}

class PropertyFindingAgent:
    """Agent for U.S. property search and analysis"""
    
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-4o"):
        self.agent = Agent(
            model=OpenAIChat(id=model_id, api_key=openai_api_key),
            markdown=True,
            description="I am a U.S. real estate expert analyzing property markets"
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    @lru_cache(maxsize=100)
    def cached_search(self, params: dict):
        """Cached property search"""
        return self.firecrawl.extract(**params)

    def parse_location(self, location: str) -> tuple:
        """Parse city and state from location input"""
        if ',' in location:
            city, state = map(str.strip, location.split(','))
            return city, state.upper()
        return location, None

    def get_regional_data(self, state: str) -> dict:
        """Get regional market data"""
        return REGIONAL_SETTINGS.get(state, {})

    def find_properties(
        self, 
        location: str,
        max_price: float,
        property_type: str = "Single-Family Home",
        bedrooms: int = None
    ) -> str:
        """Find and analyze properties in the U.S. market"""
        city, state = self.parse_location(location)
        regional_data = self.get_regional_data(state)
        
        formatted_location = f"{city.lower().replace(' ', '-')}-{state.lower()}" if state else city.lower().replace(' ', '-')
        
        urls = [
            f"https://www.zillow.com/homes/{formatted_location}_rb/",
            f"https://www.realtor.com/realestateandhomes-search/{formatted_location}",
            f"https://www.redfin.com/city/34701/{state}/{formatted_location}",
            f"https://www.trulia.com/{formatted_location}"
        ]
        
        search_prompt = f"""
        Extract residential properties matching:
        - Location: {location}
        - Max Price: ${max_price:,.0f}
        - Property Type: {property_type}
        {"- Bedrooms: " + str(bedrooms) if bedrooms else ""}
        
        Requirements:
        - Include active listings only
        - Minimum 3 properties, maximum 10
        - Include full address and key details
        - Exclude foreclosures and auctions
        """
        
        raw_response = self.cached_search({
            'urls': urls,
            'params': {
                'prompt': search_prompt,
                'schema': PropertiesResponse.model_json_schema()
            }
        })
        
        properties = raw_response['data'].get('properties', []) if raw_response.get('success') else []
        
        analysis = self.agent.run(
            f"""As a U.S. real estate expert, analyze these properties:

            Properties:
            {properties}

            **Analysis Requirements:**
            1. Market Comparison:
               - Price vs local median (${regional_data.get('price_per_sqft', 0)}/sqft)
               - Days on market analysis
            2. Property Evaluation:
               - Price/square foot
               - Year built significance
               - Lot size value
            3. Financial Considerations:
               - Tax estimates ({regional_data.get('tax_rate', 0)%} rate)
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
            """
        )
        
        return analysis.content

    def get_market_trends(self, location: str) -> str:
        """Get detailed market trends analysis"""
        city, state = self.parse_location(location)
        
        raw_response = self.cached_search({
            'urls': [f"https://www.zillow.com/{city}-{state}/housing-market/"],
            'params': {
                'prompt': f"""
                Extract market trends for {city}, {state}:
                - Neighborhood price data
                - School ratings
                - Price trends (YoY)
                - Days on market
                - Rental yields
                """,
                'schema': LocationsResponse.model_json_schema()
            }
        })
        
        if raw_response.get('success'):
            trends = raw_response['data'].get('trends', [])
            
            analysis = self.agent.run(
                f"""Analyze real estate trends for {location}:

                Data:
                {trends}

                **Required Analysis:**
                1. Price Trends by Neighborhood
                2. School District Impact
                3. Emerging Areas
                4. Investment Recommendations

                **Format:**
                
                📊 MARKET OVERVIEW
                • Current Median Price
                • Market Temperature
                
                🏫 SCHOOL IMPACT
                • Top Districts
                • Price Premiums
                
                🚀 GROWTH AREAS
                • Upcoming Neighborhoods
                • Development Projects
                
                💡 INVESTOR GUIDE
                • Best Rental Yields
                • Flipping Opportunities
                """
            )
            return analysis.content
            
        return "No market trends data available"

def create_property_agent():
    """Initialize agent with API keys"""
    if 'property_agent' not in st.session_state:
        st.session_state.property_agent = PropertyFindingAgent(
            firecrawl_api_key=st.session_state.firecrawl_key,
            openai_api_key=st.session_state.openai_key,
            model_id=st.session_state.model_id
        )

def main():
    st.set_page_config(
        page_title="US Real Estate AI Agent",
        page_icon="🏡",
        layout="wide"
    )

    with st.sidebar:
        st.title("⚙️ Configuration")
        st.session_state.model_id = st.selectbox(
            "AI Model",
            options=["gpt-4o", "gpt-4-turbo"],
            index=0
        )
        
        st.divider()
        
        st.session_state.firecrawl_key = st.text_input(
            "Firecrawl API Key",
            type="password"
        )
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key",
            type="password"
        )
        
        st.warning("""
        **Data Disclaimer:**  
        Listings are sourced from public websites.  
        Always verify with a licensed professional.
        """)

    st.title("🏡 AI U.S. Real Estate Assistant")
    st.caption("Market Analysis & Property Recommendations")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        location = st.text_input(
            "Location",
            placeholder="Enter city, state (e.g., Austin, TX)",
            help="Include state abbreviation for better results"
        )
        
        property_type = st.selectbox(
            "Property Type",
            options=["Single-Family Home", "Condo", "Townhouse", "Multi-Family"]
        )

    with col2:
        max_price = st.number_input(
            "Max Price (USD)",
            min_value=50000,
            max_value=5000000,
            value=450000,
            step=25000
        )
        
        bedrooms = st.selectbox(
            "Bedrooms",
            options=["Any", 1, 2, 3, 4, 5],
            index=0
        )

    if st.button("🚀 Analyze Properties", use_container_width=True):
        if not all([st.session_state.firecrawl_key, st.session_state.openai_key]):
            st.error("Missing API keys")
            return
            
        create_property_agent()
        
        try:
            with st.spinner("🔍 Scanning listings..."):
                properties = st.session_state.property_agent.find_properties(
                    location=location,
                    max_price=max_price,
                    property_type=property_type,
                    bedrooms=bedrooms if bedrooms != "Any" else None
                )
                
                st.markdown("---")
                st.subheader("📊 Property Analysis")
                st.markdown(properties)
                
                with st.expander("📈 Detailed Market Trends"):
                    trends = st.session_state.property_agent.get_market_trends(location)
                    st.markdown(trends)

                st.markdown("---")
                st.subheader("📋 Key Considerations")
                st.markdown("""
                - Verify property details with local MLS
                - Consult mortgage broker for financing
                - Review HOA covenants
                - Conduct professional inspection
                """)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()