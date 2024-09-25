import requests

# The URL of the FastAPI endpoint
url = 'http://127.0.0.1:8000/predict/'


def get_prediction(text):
    # The data to send to the model (text input from the user)
    data = {'text': text}
    
    # Send the POST request to the API
    response = requests.post(url, json=data)
    
    # Handle the API response
    if response.status_code == 200:
        return response.json()  # Return the prediction
    else:
        return f"Error: {response.status_code}, {response.text}"

# Interactive loop to keep asking for user input
while True:
    # Get text input from the user
    user_input = input("Enter text for prediction (or type 'exit' to quit): ")

    # If the user types 'exit', break the loop and stop the script
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
    
    # Get prediction from the API
    prediction = get_prediction(user_input)
    
    # Print the prediction result
    print("Prediction:", prediction)