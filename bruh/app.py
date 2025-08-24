from flask import Flask, render_template, jsonify
import time
test = 0

app = Flask(__name__)

# This function is not called by the script, but by a user visiting the URL
@app.route('/')
def home():
    # Imagine you fetch the initial data here
    notification_count = 5 
    return render_template('index.html', initial_notifications=notification_count)

# This is the function that the JavaScript will call
@app.route('/notifications')
def get_notifications():
    global test
    # Simulate fetching new data from a database
    time.sleep(1) # simulate network delay
    test +=5
    new_count = test
    return jsonify({'count': new_count})

if __name__ == '__main__':
    app.run(debug=True)