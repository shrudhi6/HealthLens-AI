import os
import google.generativeai as genai

class AIRecommendationEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GEMINI_API_KEY. Set it in your .env file.")
        genai.configure(api_key=api_key)

    def generate_food_plan(self, detected_conditions, predictions):
        """
        Generate a personalized nutrition plan using Google Gemini
        """
        prompt = self._build_prompt(detected_conditions, predictions)

        # ✅ Updated to supported Gemini 2.5 models
        model_names = [
            "models/gemini-2.5-pro",         # Best for long reasoning & structured text
            "models/gemini-2.5-flash",       # Faster, cheaper, still accurate
            "models/gemini-pro-latest"       # Legacy fallback
        ]

        for model_name in model_names:
            try:
                print(f"🧠 Trying Gemini model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)

                # Return response safely
                if hasattr(response, "text") and response.text:
                    return response.text
                elif response.candidates:
                    return response.candidates[0].content.parts[0].text
                else:
                    return "⚠️ Gemini returned no readable text."

            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                continue

        return "❌ All Gemini models failed to generate a recommendation. Please check your API setup."

    def _build_prompt(self, detected_conditions, predictions):
        """Create contextual prompt for Gemini"""
        condition_list = ", ".join(detected_conditions)
        prompt = f"""
        You are a certified AI nutritionist in the HealthLens AI system.
        The user has been diagnosed with: {condition_list}.

        Based on these conditions, generate a *personalized Indian diet plan* that includes:
        - 🥣 3 main meals (breakfast, lunch, dinner)
        - 🍎 2 optional healthy snacks
        - 💧 Hydration and vitamin/mineral recommendations
        - 🚫 Foods to avoid
        - 💬 Short motivational message for daily health

        Keep the plan clear, structured, and friendly.
        Example format:

        **Breakfast:** …
        **Lunch:** …
        **Dinner:** …
        **Snacks:** …
        **Hydration & Tips:** …
        **Avoid:** …
        **Motivation:** …
        """

        return prompt
