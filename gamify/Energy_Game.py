import streamlit as st
import requests
import pandas as pd

# Backend API URL
API_URL = "http://127.0.0.1:8000"  # Change this if the backend URL changes

# Streamlit app title
st.title("Energy Game - Smart Home Power Monitoring")

# Step 1: Create a User or Fetch Existing User
user_name = st.text_input("Enter your username:")

if user_name:
    # Check if user already exists in the backend
    response = requests.get(f"{API_URL}/user/{user_name}")
    if response.status_code == 404:
        st.write(f"User {user_name} not found, creating new user...")
        # If not found, create a new user
        data = {
            "username": user_name,
            "points": 0,
            "energy_usage": 500,
            "last_challenge_completed": None
        }
        response = requests.post(f"{API_URL}/user/", json=data)
        user_data = response.json()
        st.write(f"New user created: {user_data['username']}! Your current points: {user_data['points']}")
    else:
        # If the user exists, fetch their data
        user_data = response.json()
        st.write(f"Welcome back, {user_data['username']}! Your current points: {user_data['points']}")

# Step 2: Show Current Energy Usage
if user_name:
    response = requests.get(f"{API_URL}/user/{user_name}")
    user_data = response.json()
    st.write(f"Current energy usage: {user_data['energy_usage']} kWh")

# Step 3: Fetch and Display a Challenge
if user_name:
    try:
        response = requests.get(f"{API_URL}/challenge/", params={"username": user_name})
        if response.status_code == 200:
            challenge_data = response.json()
            if "challenge" in challenge_data:
                challenge = challenge_data['challenge']
                
                st.write(f"Challenge: {challenge['name']}")
                st.write(f"Points for completion: {challenge['points']}")
            elif "message" in challenge_data:
                st.info(challenge_data["message"])
            else:
                st.warning("Unexpected response from server.")
        else:
            st.error("Failed to fetch challenge. Please try again later.")
    except Exception as e:
        st.error(f"Error fetching challenge: {e}")

# Step 4: Complete Challenge Button
if user_name and 'challenge' in locals():
    if st.button("Complete Challenge"):
        # Completing the challenge and updating user points
        challenge_data = {
            "name": challenge['name'],
            "points": challenge['points']
        }
        response = requests.post(f"{API_URL}/user/{user_name}/complete_challenge", json=challenge_data)
        updated_user = response.json()

        if "user" in updated_user:
            st.success("Challenge Completed!")
            st.write(f"New points: {updated_user['user']['points']}")
            st.write(f"Last challenge: {updated_user['user'].get('last_challenge_completed', 'N/A')}")

            # Display updated user data and history
            user_history = updated_user['user'].get('history', [])
            if user_history:
                st.write("Challenge History:")
                history_df = pd.DataFrame(user_history)
                st.dataframe(history_df)
            else:
                st.write("No history available.")

            # Show new challenge if available
            if "new_challenge" in updated_user:
                new_chal = updated_user["new_challenge"]
                
                # Display the new random challenge
                st.subheader("🎯 New Random Challenge")
                st.write(f"**{new_chal['name']}**")
                st.write(f"Points: {new_chal['points']}")
                
                # Update the challenge variable to the new random challenge
                challenge = new_chal
            else:
                # If there are no more new challenges, show a message
                st.write("No more challenges available! Keep up the great work!")

        else:
            st.error(updated_user.get("message", "Unexpected response from server."))


# Step 5: Visualize User Data
if user_name:
    response = requests.get(f"{API_URL}/user/{user_name}")
    user_data = response.json()
    data = {
        'User': [user_data['username']],
        'Points': [user_data['points']],
        'Energy Usage': [user_data['energy_usage']]
    }
    df = pd.DataFrame(data)
    st.write("User Data", df)

# Step 6: Add More Features (for example, show tips for energy saving)
st.sidebar.header("Energy Saving Tips")
st.sidebar.write("1. Turn off lights when not in use.")
st.sidebar.write("2. Use energy-efficient appliances.")
st.sidebar.write("3. Install a smart thermostat.")
st.sidebar.write("4. Reduce standby power consumption by unplugging devices.")

# streamlit run "C:\Users\Akash\Desktop\electricity3\gamify\Energy_Game.py"