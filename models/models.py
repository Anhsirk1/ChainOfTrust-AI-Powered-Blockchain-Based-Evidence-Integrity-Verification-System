import uuid
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    # NEW STRUCTURE
    department = db.Column(db.String(50), nullable=False)   # police / forensic / court / prosecution
    designation = db.Column(db.String(100), nullable=False) # SI / DSP / Judge / etc
    is_admin = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.fullname} | {self.department} | Admin={self.is_admin}>'
    
# ================= FIR MODEL =================
class FIR(db.Model):
    __tablename__ = 'firs'

    id = db.Column(db.Integer, primary_key=True)
    fir_number = db.Column(db.String(50), unique=True, nullable=False)

    # ===== Complainant Details =====
    complainant_name = db.Column(db.String(120), nullable=False)
    guardian_type = db.Column(db.String(20))
    guardian_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    marital_status = db.Column(db.String(20))
    occupation = db.Column(db.String(100))
    aadhaar = db.Column(db.String(12))
    mobile = db.Column(db.String(15))
    address = db.Column(db.Text)

    # ===== Incident Details =====
    incident_date = db.Column(db.DateTime, nullable=False)
    incident_time = db.Column(db.String(10))
    location = db.Column(db.String(200), nullable=False)
    police_station = db.Column(db.String(150))
    offence_nature = db.Column(db.String(50))
    sections = db.Column(db.String(200), nullable=False)

    # ===== Accused Details =====
    accused_known = db.Column(db.String(10))
    accused_name = db.Column(db.String(120))
    accused_address = db.Column(db.Text)

    # ===== Description =====
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="Active", nullable=False)
    police_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ================= EVIDENCE MODEL =================
class Evidence(db.Model):
    __tablename__ = 'evidences'

    id = db.Column(db.Integer, primary_key=True)

    # 🔗 FIR & Officer
    fir_id = db.Column(db.Integer, db.ForeignKey('firs.id'), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # 🏷 Evidence Info
    title = db.Column(db.String(200), nullable=False)
    evidence_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)

    # 🏢 Facility & Storage
    facility_name = db.Column(db.String(150), nullable=False)
    collection_room = db.Column(db.String(100), nullable=False)
    storage_type = db.Column(db.String(50), nullable=False)
    storage_unit = db.Column(db.String(50))
    storage_slot = db.Column(db.String(50))

    # 🔐 Evidence Seal (AUTO)
    evidence_seal_id = db.Column(db.String(100), unique=True, nullable=False)

    # 🔐 Sensitivity
    sensitivity = db.Column(db.String(30), default="Normal")

    # 📂 File Details
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))

    # 🔐 Integrity
    file_hash = db.Column(db.String(256), nullable=False, unique=True)

    # 🌐 Digital Footprint
    upload_ip = db.Column(db.String(45))
    user_agent = db.Column(db.Text)

    # 🧪 Status
    forensic_status = db.Column(db.String(20), default="Pending")
    blockchain_status = db.Column(db.String(20), default="Pending")

    # ⏱ Timestamp (AUTO)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # 🔁 Relationships (KEEP)
    fir = db.relationship('FIR', backref=db.backref('evidences', lazy=True))
    officer = db.relationship('User', backref=db.backref('uploaded_evidences', lazy=True))

    def __repr__(self):
        return f"<Evidence {self.id} | FIR {self.fir_id} | {self.title}>"


# ================= AUDIT LOG =================
class EvidenceAudit(db.Model):
    __tablename__ = 'evidence_audits'

    id = db.Column(db.Integer, primary_key=True)
    evidence_id = db.Column(db.Integer, db.ForeignKey('evidences.id'), nullable=False)
    action = db.Column(db.String(200), nullable=False)
    actor_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# ================= MOVEMENT LOG =================
class EvidenceMovement(db.Model):
    __tablename__ = 'evidence_movements'

    id = db.Column(db.Integer, primary_key=True)
    evidence_id = db.Column(db.Integer, db.ForeignKey('evidences.id'), nullable=False)
    from_location = db.Column(db.String(200))
    to_location = db.Column(db.String(200))
    reason = db.Column(db.Text)
    moved_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    moved_at = db.Column(db.DateTime, default=datetime.utcnow)
