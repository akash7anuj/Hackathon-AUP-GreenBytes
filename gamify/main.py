from flask import Flask, jsonify, request, abort
import random
import json
import os

from datetime import datetime

app = Flask(__name__)

# Path to the JSON file for storing user data and history
USER_DATA_FILE = r"gamify/users.json"

# Helper function to load user data from JSON file
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    return {}  # Return empty dict if the file does not exist

# Helper function to save user data to JSON file
def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Load data into memory
fake_users = load_user_data()

# Simulated challenges
fake_challenges = [
    {"name": "Reduce energy by 10% this week", "points": 50},
    {"name": "Use only energy-efficient appliances today", "points": 30},
    {"name": "Install a smart thermostat", "points": 70},
    {"name": "Reduce energy usage by 20% this month", "points": 100},
]

# Helper function to get a new challenge (excluding previously completed challenges)
def get_new_challenge(user):
    completed_challenges = set(user['completed_challenges'])
    available_challenges = [ch for ch in fake_challenges if ch['name'] not in completed_challenges]
    if available_challenges:
        return random.choice(available_challenges)
    else:
        return None  # No new challenges available, all challenges completed

# Root route
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Energy Game API!"})

# Get user info
@app.route("/user/<username>", methods=["GET"])
def get_user(username):
    user = fake_users.get(username)
    if not user:
        abort(404, description="User not found")
    return jsonify(user)

# Create a new user
@app.route("/user/", methods=["POST"])
def create_user():
    data = request.json
    if not data or "username" not in data:
        abort(400, description="Missing required fields")

    username = data["username"]
    if username in fake_users:
        return jsonify({"message": "User already exists"}), 400

    fake_users[username] = {
        "username": username,
        "points": 0,
        "energy_usage": 500,
        "last_challenge_completed": None,
        "completed_challenges": [],
        "history": []  # List to track all completed challenges over time
    }
    save_user_data(fake_users)
    return jsonify(fake_users[username]), 201

@app.route("/challenge/", methods=["GET"])
def get_challenge():
    username = request.args.get("username")
    if not username:
        return jsonify({"message": "Username is required"}), 400

    user = next((u for u in fake_users.values() if u["username"] == username), None)
    if not user:
        return jsonify({"message": f"User {username} not found"}), 404

    # Get list of challenges not yet completed by the user
    completed = user.get("history", [])
    available = [c for c in fake_challenges if c["name"] not in [h["name"] for h in completed]]

    if not available:
        return jsonify({"message": "All challenges completed! Come back later for new ones."})

    challenge = random.choice(available)
    return jsonify({"challenge": challenge})

@app.route("/user/<username>/complete_challenge", methods=["POST"])
def complete_challenge(username):
    user = fake_users.get(username)
    if not user:
        abort(404, description="User not found")

    # Ensure the user has necessary fields
    user.setdefault("completed_challenges", [])
    user.setdefault("history", [])

    data = request.json
    if not data or "name" not in data or "points" not in data:
        abort(400, description="Invalid challenge data")

    challenge_name = data["name"]
    if challenge_name in user["completed_challenges"]:
        return jsonify({"message": "You have already completed this challenge!"}), 400

    # Update user points and challenge history
    user["points"] += data["points"]
    user["completed_challenges"].append(challenge_name)
    user["last_challenge_completed"] = challenge_name
    user["history"].append({
        "challenge": challenge_name,
        "points": data["points"],
        "completed_at": datetime.now().isoformat()
    })

    # Save updated data to JSON file
    save_user_data(fake_users)

    
    # Randomly select a new uncompleted challenge
    remaining_challenges = [
        ch for ch in fake_challenges if ch["name"] not in user["completed_challenges"]
    ]
    new_challenge = random.choice(remaining_challenges) if remaining_challenges else None

    if new_challenge:
        return jsonify({
            "message": "Challenge completed!",
            "new_challenge": new_challenge,
            "user": user
        })
    else:
        return jsonify({
            "message": "You have completed all challenges! 🎉",
            "user": user
        })



if __name__ == "__main__":
    app.run(port=8000, debug=True)
