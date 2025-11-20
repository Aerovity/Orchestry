"""
Test script to verify Gemini API key is working and check quota limits.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test Gemini API key and check limits."""

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")

    print("=" * 70)
    print("GEMINI API KEY TEST")
    print("=" * 70)

    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file!")
        print("\nPlease add to .env file:")
        print("GEMINI_API_KEY=your-api-key-here")
        return False

    print(f"✅ API Key found: {api_key[:10]}...{api_key[-10:]}")
    print()

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Test with a simple request
        print("Testing API with simple request...")
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")

        response = model.generate_content(
            "Say 'Hello! API is working!' in exactly 5 words.",
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )

        print(f"✅ API Response: {response.text}")
        print()

        # Check quota info
        print("=" * 70)
        print("API KEY STATUS")
        print("=" * 70)
        print("✅ API Key: VALID and WORKING")
        print("✅ Model: gemini-2.0-flash-thinking-exp")
        print("✅ Request: SUCCESS")
        print()

        # Try to get quota info (if available)
        try:
            print("Checking quota limits...")
            print("Note: Quota limits are shown in Google Cloud Console")
            print("      Go to: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
        except Exception as e:
            print(f"Could not fetch quota info automatically: {e}")

        print()
        print("=" * 70)
        print("TIER DETECTION")
        print("=" * 70)
        print("To verify if you're on PAID tier:")
        print("1. Go to: https://console.cloud.google.com/billing")
        print("2. Check 'Billing Account' is active")
        print("3. Check 'Credits' section shows $300")
        print("4. Go to: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
        print("5. Look for 'Generate requests per minute per project per model'")
        print("   - Free tier: 10-15 RPM")
        print("   - Paid tier: 1000 RPM")
        print()

        return True

    except Exception as e:
        print(f"❌ ERROR: API request failed!")
        print(f"Error: {e}")
        print()
        print("Common issues:")
        print("1. API key is invalid or expired")
        print("2. Generative AI API not enabled in Google Cloud")
        print("3. Billing not activated (even with free credit)")
        print()
        print("To fix:")
        print("1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
        print("2. Click 'ENABLE'")
        print("3. Go to: https://console.cloud.google.com/billing")
        print("4. Verify billing account is linked")
        return False


if __name__ == "__main__":
    success = test_gemini_api()

    if success:
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Your Gemini API is working correctly.")
        print("You can now run training with:")
        print()
        print("  python main.py --mode research --episodes 2 --question 'test' --use-llm-judge --verbose")
        print()
    else:
        print("=" * 70)
        print("❌ TESTS FAILED")
        print("=" * 70)
        print("Please fix the issues above before running training.")
