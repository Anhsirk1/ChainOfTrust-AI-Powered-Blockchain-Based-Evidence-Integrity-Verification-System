from flask import Flask, render_template
from models.models import db, User
from controllers.controllers import controllers
from werkzeug.security import generate_password_hash

app = Flask(__name__)

# ------------------- FLASK CONFIG -------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///evidence_chain.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'securekey'

# Initialize Database & Controllers
db.init_app(app)
app.register_blueprint(controllers)

# ------------------- DEFAULT ADMIN CREATION -------------------
def create_admin_user():
    admin_email = 'admin@chainsecure.com'
    admin = User.query.filter_by(email=admin_email).first()

    if not admin:
        admin = User(
            fullname='System Admin',
            email=admin_email,
            password=generate_password_hash('admin123', method='pbkdf2:sha256'),
            department='admin',
            designation='System Administrator',
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin created (admin@chainsecure.com | admin123)")
    else:
        print("ℹ️ Admin already exists")


# ------------------- ROUTES (ALL KEPT) -------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')


@app.route('/police_login')
def police_login():
    return render_template('police_login.html')


@app.route('/forensic_login')
def forensic_login():
    return render_template('forensic_login.html')

@app.route('/judiciary_login')
def judiciary_login():
    return render_template('judiciary_login.html')


# ------------------- APP STARTUP -------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        create_admin_user()

    app.run(debug=True)
