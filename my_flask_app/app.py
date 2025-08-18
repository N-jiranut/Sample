from flask import Flask, render_template, request

app = Flask(__name__)

# ✅ Your prebuilt function
def my_algorithm(x):
    return int(x)  # example: cube the number

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    if request.method == "POST":
        # user_input = request.form["user_input"]
        try:
            # result = my_algorithm(user_input)
            result="test"
        except Exception as e:
            error = str(e)
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
