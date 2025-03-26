from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import hashlib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'your_username'  # Replace with your username
app.config['MYSQL_PASSWORD'] = 'your_password'  # Replace with your password
app.config['MYSQL_DB'] = 'your_database'  # Replace with your database name
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Document upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist

mysql = MySQL(app)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        hashed_password = hash_password(password)
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, hashed_password))
        user = cur.fetchone()
        cur.close()
        if user:
            session['loggedin'] = True
            session['id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username or password!'
    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        hashed_password = hash_password(password)
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cur.fetchone()
        if account:
            msg = 'Account already exists!'
        else:
            cur.execute('INSERT INTO users (username, password, email) VALUES (%s, %s, %s)', (username, hashed_password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)

@app.route('/home', methods=['GET','POST'])
def home():
    if 'loggedin' in session:
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM grades WHERE user_id = %s', (session['id'],))
        grades = cur.fetchall()
        cur.close()

        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Optionally, store file path in database for retrieval
                return redirect(url_for('home'))

        return render_template('home.html', username=session['username'], grades=grades)
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'loggedin' in session:
        cur = mysql.connection.cursor()
        cur.execute('SELECT * FROM users WHERE id = %s', (session['id'],))
        user = cur.fetchone()

        msg = ''

        if request.method == 'POST':
            new_email = request.form.get('email')
            new_password = request.form.get('password')

            if new_email:
                cur.execute('UPDATE users SET email = %s WHERE id = %s', (new_email, session['id']))
                mysql.connection.commit()
                msg = "Email updated successfully"
            if new_password:
                hashed_new_password = hash_password(new_password)
                cur.execute('UPDATE users SET password = %s WHERE id = %s', (hashed_new_password, session['id']))
                mysql.connection.commit()
                msg = "Password updated successfully"

        cur.execute('SELECT * FROM users WHERE id = %s', (session['id'],))
        user = cur.fetchone()
        cur.close()

        return render_template('profile.html', user=user, msg=msg)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)