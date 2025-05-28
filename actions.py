import requests
from rasa_sdk import Action
from rasa_sdk.events import SlotSet

class ActionFetchMedicalInfo(Action):
    def name(self):
        return "action_fetch_medical_info"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")

        # Send request to AI chatbot API (Ollama)
        response = requests.post("http://127.0.0.1:5000/chatbot", json={"message": user_message})
        
        if response.status_code == 200:
            ai_response = response.json().get("response", "Sorry, I couldn't fetch that info.")
        else:
            ai_response = "Sorry, there was an issue retrieving the medical information."

        dispatcher.utter_message(text=ai_response)
        return []

class ActionBookAppointment(Action):
    def name(self):
        return "action_book_appointment"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(text="Please provide your preferred date and time for the appointment.")
        return []

class ActionHealthAdvice(Action):
    def name(self):
        return "action_health_advice"

    def run(self, dispatcher, tracker, domain):
        advice = [
            "Drink plenty of water and maintain a healthy diet.",
            "Exercise at least 30 minutes a day for good heart health.",
            "Reduce stress through meditation and proper sleep.",
            "Limit sugar intake to maintain balanced blood glucose levels.",
            "Wash your hands frequently to prevent infections."
        ]
        dispatcher.utter_message(text=random.choice(advice))
        return []
